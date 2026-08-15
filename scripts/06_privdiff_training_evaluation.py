#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ==========================================================
# Cell 1 - PrivDiffuser Training Configuration
# CPU and Narval/GPU Compatible
# ==========================================================

import os
import sys
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
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ----------------------------------------------------------
# Run mode
# ----------------------------------------------------------
# small_cpu:
#     Small local experiment.
#
# full_gpu:
#     Full multi-subject Narval experiment.
#
# On Narval:
#     export RUN_MODE=full_gpu
# ----------------------------------------------------------

RUN_MODE = os.environ.get(
    "RUN_MODE",
    "small_cpu"
).lower()

if RUN_MODE not in {
    "small_cpu",
    "full_gpu"
}:
    raise ValueError(
        "RUN_MODE must be either "
        "'small_cpu' or 'full_gpu'."
    )


# ----------------------------------------------------------
# Project root
# ----------------------------------------------------------
# Local default:
#     C:\PrivDiffuser_Narval
#
# On Narval, define:
#     export PRIVDIFFUSER_ROOT=/path/to/PrivDiffuser_Narval
# ----------------------------------------------------------

PRIVDIFFUSER_ROOT = Path(
    os.environ.get(
        "PRIVDIFFUSER_ROOT",
        r"C:\PrivDiffuser_Narval"
    )
).expanduser().resolve()

if not PRIVDIFFUSER_ROOT.exists():
    raise FileNotFoundError(
        "PrivDiffuser project folder was not found:\n"
        f"{PRIVDIFFUSER_ROOT}"
    )


# ----------------------------------------------------------
# Working directory
# ----------------------------------------------------------
# Contains saved checkpoints and experimental results.
# ----------------------------------------------------------

WORK_DIR = Path(
    os.environ.get(
        "PRIVDIFFUSER_WORK_DIR",
        str(
            PRIVDIFFUSER_ROOT
            / "new_dataset"
        )
    )
).expanduser().resolve()

if not WORK_DIR.exists():
    raise FileNotFoundError(
        "Working directory was not found:\n"
        f"{WORK_DIR}"
    )


# ----------------------------------------------------------
# Processed dataset directory
# ----------------------------------------------------------
# Local processed data:
# C:\PrivDiffuser_Narval\datasets\
# DatasetIMUandBIOMARKERS\processed
#
# On Narval, define:
# export IMU_PROCESSED_PATH=/path/to/processed
# ----------------------------------------------------------

PROCESSED_DIR = Path(
    os.environ.get(
        "IMU_PROCESSED_PATH",
        str(
            PRIVDIFFUSER_ROOT
            / "datasets"
            / "DatasetIMUandBIOMARKERS"
            / "processed"
        )
    )
).expanduser().resolve()

if not PROCESSED_DIR.exists():
    raise FileNotFoundError(
        "Processed dataset directory was not found:\n"
        f"{PROCESSED_DIR}"
    )


# ----------------------------------------------------------
# Add original PrivDiffuser source to Python path
# ----------------------------------------------------------

project_root_string = str(
    PRIVDIFFUSER_ROOT
)

if project_root_string not in sys.path:
    sys.path.insert(
        0,
        project_root_string
    )


# ----------------------------------------------------------
# Dataset dimensions
# ----------------------------------------------------------

WINDOW_SIZE = 128
NUM_FEATURES = 30

NUM_ACTIVITIES = 6
NUM_CLASSES = NUM_ACTIVITIES

NUM_GENDERS = 2
NUM_WEIGHT_CLASSES = 3


# ----------------------------------------------------------
# Utility and private attributes
# ----------------------------------------------------------

UTILITY_ATTRIBUTE = "activity"
PRIVATE_ATTRIBUTE = "gender"

if PRIVATE_ATTRIBUTE == "gender":
    NUM_PRIVATE_CLASSES = NUM_GENDERS

elif PRIVATE_ATTRIBUTE == "weight":
    NUM_PRIVATE_CLASSES = NUM_WEIGHT_CLASSES

else:
    raise ValueError(
        "PRIVATE_ATTRIBUTE must be either "
        "'gender' or 'weight'."
    )


# ----------------------------------------------------------
# Latent representation size
# ----------------------------------------------------------

Z_DIM = 60


# ----------------------------------------------------------
# Training configuration
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    BATCH_SIZE = 8
    NUM_WORKERS = 0

    DIFFUSION_EPOCHS = 5
    DIFFUSION_STEPS = 50

    LEARNING_RATE = 0.001

else:

    BATCH_SIZE = 16
    NUM_WORKERS = int(
        os.environ.get(
            "NUM_WORKERS",
            "2"
        )
    )

    DIFFUSION_EPOCHS = 30
    DIFFUSION_STEPS = 1000

    LEARNING_RATE = 0.0002


# ----------------------------------------------------------
# Checkpoint paths
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    SURROGATE_CHECKPOINT = (
        WORK_DIR
        / "small_surrogate_activity_model.pt"
    )

    PRIVACY_CHECKPOINT = (
        WORK_DIR
        / "small_cpu_privacy_gender_model.pt"
    )

    DIFFUSION_CHECKPOINT = (
        WORK_DIR
        / "small_conditional_denoiser.pt"
    )

else:

    SURROGATE_CHECKPOINT = (
        WORK_DIR
        / "full_surrogate_activity_model.pt"
    )

    PRIVACY_CHECKPOINT = (
        WORK_DIR
        / "full_gpu_privacy_gender_model.pt"
    )

    DIFFUSION_CHECKPOINT = (
        WORK_DIR
        / "full_conditional_denoiser.pt"
    )


# ----------------------------------------------------------
# Check required preliminary checkpoints
# ----------------------------------------------------------

if not SURROGATE_CHECKPOINT.exists():
    raise FileNotFoundError(
        "Surrogate model checkpoint was not found:\n"
        f"{SURROGATE_CHECKPOINT}"
    )

if not PRIVACY_CHECKPOINT.exists():
    raise FileNotFoundError(
        "Privacy model checkpoint was not found:\n"
        f"{PRIVACY_CHECKPOINT}"
    )


# ----------------------------------------------------------
# Count processed files
# ----------------------------------------------------------

TRAIN_FILE_COUNT = len(
    list(
        PROCESSED_DIR.glob(
            "*_train.npz"
        )
    )
)

TEST_FILE_COUNT = len(
    list(
        PROCESSED_DIR.glob(
            "*_test.npz"
        )
    )
)

if TRAIN_FILE_COUNT == 0:
    raise FileNotFoundError(
        "No processed training files were found in:\n"
        f"{PROCESSED_DIR}"
    )

if TEST_FILE_COUNT == 0:
    raise FileNotFoundError(
        "No processed testing files were found in:\n"
        f"{PROCESSED_DIR}"
    )


# ----------------------------------------------------------
# Configuration summary
# ----------------------------------------------------------

print("=" * 75)
print("PRIVDIFFUSER TRAINING CONFIGURATION")
print("=" * 75)

print("Run mode              :", RUN_MODE)
print("Device                :", device)

print("\nProject root          :", PRIVDIFFUSER_ROOT)
print("Working directory     :", WORK_DIR)
print("Processed directory   :", PROCESSED_DIR)

print("\nTraining files        :", TRAIN_FILE_COUNT)
print("Testing files         :", TEST_FILE_COUNT)

print("\nUtility attribute     :", UTILITY_ATTRIBUTE)
print("Private attribute     :", PRIVATE_ATTRIBUTE)

print("\nActivity classes      :", NUM_ACTIVITIES)
print("Private classes       :", NUM_PRIVATE_CLASSES)
print("Embedding dimension   :", Z_DIM)

print("\nWindow size           :", WINDOW_SIZE)
print("Sensor features       :", NUM_FEATURES)

print("\nBatch size            :", BATCH_SIZE)
print("DataLoader workers    :", NUM_WORKERS)
print("Diffusion epochs      :", DIFFUSION_EPOCHS)
print("Diffusion steps       :", DIFFUSION_STEPS)
print("Learning rate         :", LEARNING_RATE)

print("\nSurrogate checkpoint  :", SURROGATE_CHECKPOINT)
print("Privacy checkpoint    :", PRIVACY_CHECKPOINT)
print("Diffusion checkpoint  :", DIFFUSION_CHECKPOINT)

print(
    "\nConfiguration completed successfully."
)


# In[9]:


# ==========================================================
# Cell 2 - Import Core Diffusion Modules
# ==========================================================

import sys
import importlib
from pathlib import Path


print("=" * 75)
print("IMPORTING CORE DIFFUSION MODULES")
print("=" * 75)


# ----------------------------------------------------------
# Clean paths accidentally added by earlier import attempts
# ----------------------------------------------------------

incorrect_directories = {
    str(PRIVDIFFUSER_ROOT / "mine"),
    str(PRIVDIFFUSER_ROOT / "unet"),
    str(PRIVDIFFUSER_ROOT / "diffusion"),
    str(PRIVDIFFUSER_ROOT / "embedding"),
}

sys.path = [
    path
    for path in sys.path
    if str(Path(path)) not in incorrect_directories
]


# ----------------------------------------------------------
# Ensure only the project root is added
# ----------------------------------------------------------

project_root_string = str(PRIVDIFFUSER_ROOT)

if project_root_string in sys.path:
    sys.path.remove(project_root_string)

sys.path.insert(0, project_root_string)


# ----------------------------------------------------------
# Remove modules cached during failed imports
# ----------------------------------------------------------

modules_to_remove = [
    module_name
    for module_name in list(sys.modules.keys())
    if (
        module_name == "mine"
        or module_name.startswith("mine.")
    )
]

for module_name in modules_to_remove:
    del sys.modules[module_name]

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
print("MINE will be configured later for privacy guidance.")


# In[11]:


# ==========================================================
# Cell 3 - Preloaded IMU Dataset for Diffusion Training
# CPU and Narval/GPU Compatible
# ==========================================================

import gc
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


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
            f"DIFFUSION DATASET"
        )
        print("=" * 75)

        # --------------------------------------------------
        # Load each subject file ONCE
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

        # Release temporary lists.
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
        # Summary
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

        # --------------------------------------------------
        # IMPORTANT:
        # No np.load() occurs here.
        # --------------------------------------------------

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
print("DIFFUSION DATASET AND DATALOADER CHECK")
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


# In[12]:


# ==========================================================
# Cell 4 - Load Surrogate Utility Classifier
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallSurrogateClassifier(nn.Module):
    """
    Lightweight activity classifier used to measure
    preservation of the public/utility attribute.

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

        self.embedding = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=z_dim
            ),
            nn.ReLU()
        )

        self.classifier = nn.Linear(
            in_features=z_dim,
            out_features=num_classes
        )

    def forward(self, x):

        features = self.feature_extractor(x)

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
# Create model
# ----------------------------------------------------------

surrogate_model = SmallSurrogateClassifier(
    num_classes=NUM_ACTIVITIES,
    z_dim=Z_DIM
).to(device)


# ----------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------

checkpoint = torch.load(
    SURROGATE_CHECKPOINT,
    map_location=device,
    weights_only=False
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    surrogate_state_dict = checkpoint[
        "model_state_dict"
    ]

else:
    surrogate_state_dict = checkpoint


surrogate_model.load_state_dict(
    surrogate_state_dict
)


# ----------------------------------------------------------
# Freeze model
# ----------------------------------------------------------

surrogate_model.eval()

for parameter in surrogate_model.parameters():
    parameter.requires_grad = False


# ----------------------------------------------------------
# Test model on one batch
# ----------------------------------------------------------

with torch.no_grad():

    sample_windows_device = (
        sample_windows.to(device)
    )

    activity_logits, z_public = (
        surrogate_model(
            sample_windows_device
        )
    )

    activity_predictions = (
        activity_logits.argmax(dim=1)
    )


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
        for parameter in surrogate_model.parameters()
    )
)

print(
    "Logits shape       :",
    tuple(activity_logits.shape)
)

print(
    "Public embedding   :",
    tuple(z_public.shape)
)

print(
    "True activities    :",
    sample_activities.tolist()
)

print(
    "Predicted activities:",
    activity_predictions.cpu().tolist()
)

print(
    "\nSurrogate utility classifier loaded successfully."
)


# In[14]:


# ==========================================================
# Cell 5 - Load Privacy Classifier
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallPrivacyClassifier(nn.Module):
    """
    Lightweight gender classifier used as the privacy attacker.

    The layer names match the saved checkpoint exactly.

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

        # Important:
        # This name must match the saved checkpoint.
        self.private_embedding = nn.Linear(
            in_features=32,
            out_features=z_dim
        )

        self.private_classifier = nn.Linear(
            in_features=z_dim,
            out_features=num_classes
        )

    def forward(self, x):

        features = self.feature_extractor(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        z_private = F.relu(
            self.private_embedding(features)
        )

        logits = self.private_classifier(
            z_private
        )

        return logits, z_private


# ----------------------------------------------------------
# Create model
# ----------------------------------------------------------

privacy_model = SmallPrivacyClassifier(
    num_classes=NUM_PRIVATE_CLASSES,
    z_dim=Z_DIM
).to(device)


# ----------------------------------------------------------
# Load checkpoint
# ----------------------------------------------------------

checkpoint = torch.load(
    PRIVACY_CHECKPOINT,
    map_location=device,
    weights_only=False
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    privacy_state_dict = checkpoint[
        "model_state_dict"
    ]
else:
    privacy_state_dict = checkpoint


privacy_model.load_state_dict(
    privacy_state_dict
)


# ----------------------------------------------------------
# Freeze model
# ----------------------------------------------------------

privacy_model.eval()

for parameter in privacy_model.parameters():
    parameter.requires_grad = False


# ----------------------------------------------------------
# Test model on one batch
# ----------------------------------------------------------

with torch.no_grad():

    privacy_logits, z_private = (
        privacy_model(
            sample_windows_device
        )
    )

    privacy_predictions = (
        privacy_logits.argmax(dim=1)
    )


print("=" * 75)
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
        for parameter in privacy_model.parameters()
    )
)

print(
    "Logits shape        :",
    tuple(privacy_logits.shape)
)

print(
    "Private embedding   :",
    tuple(z_private.shape)
)

print(
    "True genders        :",
    sample_genders.tolist()
)

print(
    "Predicted genders   :",
    privacy_predictions.cpu().tolist()
)

print(
    "\nPrivacy classifier loaded successfully."
)


# In[15]:


# ==========================================================
# Cell 6 - Diffusion Schedule and Forward Noising
# ==========================================================

import torch


# ----------------------------------------------------------
# Linear beta schedule
# ----------------------------------------------------------

def linear_beta_schedule(
    timesteps,
    beta_start=1e-4,
    beta_end=0.02
):
    """
    Creates a linear noise schedule.

    beta_t controls how much noise is added
    at each diffusion timestep.
    """

    return torch.linspace(
        beta_start,
        beta_end,
        timesteps,
        dtype=torch.float32,
        device=device
    )


# ----------------------------------------------------------
# Diffusion coefficients
# ----------------------------------------------------------

betas = linear_beta_schedule(
    DIFFUSION_STEPS
)

alphas = 1.0 - betas

alphas_cumprod = torch.cumprod(
    alphas,
    dim=0
)

alphas_cumprod_previous = F.pad(
    alphas_cumprod[:-1],
    pad=(1, 0),
    value=1.0
)

sqrt_alphas_cumprod = torch.sqrt(
    alphas_cumprod
)

sqrt_one_minus_alphas_cumprod = torch.sqrt(
    1.0 - alphas_cumprod
)

sqrt_recip_alphas = torch.sqrt(
    1.0 / alphas
)

posterior_variance = (
    betas
    * (
        1.0 - alphas_cumprod_previous
    )
    / (
        1.0 - alphas_cumprod
    )
)


# ----------------------------------------------------------
# Extract timestep-specific coefficients
# ----------------------------------------------------------

def extract(
    coefficients,
    timesteps,
    target_shape
):
    """
    Selects one coefficient for every sample in a batch
    and reshapes it for broadcasting.
    """

    batch_size = timesteps.shape[0]

    selected = coefficients.gather(
        0,
        timesteps
    )

    return selected.reshape(
        batch_size,
        *((1,) * (len(target_shape) - 1))
    )


# ----------------------------------------------------------
# Forward diffusion q(x_t | x_0)
# ----------------------------------------------------------

def forward_diffusion_sample(
    x_0,
    timesteps,
    noise=None
):
    """
    Adds noise to clean IMU windows.

    x_t =
        sqrt(alpha_bar_t) * x_0
        +
        sqrt(1 - alpha_bar_t) * noise
    """

    if noise is None:
        noise = torch.randn_like(
            x_0
        )

    sqrt_alpha_bar_t = extract(
        sqrt_alphas_cumprod,
        timesteps,
        x_0.shape
    )

    sqrt_one_minus_alpha_bar_t = extract(
        sqrt_one_minus_alphas_cumprod,
        timesteps,
        x_0.shape
    )

    noisy_windows = (
        sqrt_alpha_bar_t * x_0
        +
        sqrt_one_minus_alpha_bar_t * noise
    )

    return noisy_windows, noise


# ----------------------------------------------------------
# Test forward diffusion on one batch
# ----------------------------------------------------------

sample_windows_device = sample_windows.to(
    device
)

sample_timesteps = torch.randint(
    low=0,
    high=DIFFUSION_STEPS,
    size=(
        sample_windows_device.shape[0],
    ),
    device=device,
    dtype=torch.long
)

noisy_sample_windows, sampled_noise = (
    forward_diffusion_sample(
        sample_windows_device,
        sample_timesteps
    )
)


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
    tuple(sample_windows_device.shape)
)

print(
    "Noisy batch shape         :",
    tuple(noisy_sample_windows.shape)
)

print(
    "Noise tensor shape        :",
    tuple(sampled_noise.shape)
)

print(
    "Sampled timesteps         :",
    sample_timesteps.cpu().tolist()
)

print(
    "First beta                :",
    float(betas[0].cpu())
)

print(
    "Last beta                 :",
    float(betas[-1].cpu())
)

print(
    "\nForward diffusion configured successfully."
)


# In[16]:


# ==========================================================
# Cell 7 - Inspect PrivDiffuser Model Interfaces
# ==========================================================

import inspect


print("=" * 75)
print("PRIVDIFFUSER CLASS INTERFACE CHECK")
print("=" * 75)


print("\nUnet constructor:")
print(
    inspect.signature(
        Unet.__init__
    )
)


print("\nConditionalEmbedding constructor:")
print(
    inspect.signature(
        ConditionalEmbedding.__init__
    )
)


print("\nGaussianDiffusion constructor:")
print(
    inspect.signature(
        GaussianDiffusion.__init__
    )
)


print("\nUnet forward method:")
print(
    inspect.signature(
        Unet.forward
    )
)


print("\nConditionalEmbedding forward method:")
print(
    inspect.signature(
        ConditionalEmbedding.forward
    )
)


print("\nGaussianDiffusion available public methods:")

diffusion_methods = [
    method_name
    for method_name, method_object in inspect.getmembers(
        GaussianDiffusion,
        predicate=inspect.isfunction
    )
    if not method_name.startswith("_")
]

print(diffusion_methods)


print(
    "\nInterface inspection completed successfully."
)


# In[23]:


# ==========================================================
# Cell 8 - Initialize Conditional U-Net Denoiser
# ==========================================================

import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Conditioning configuration
# ----------------------------------------------------------
# The surrogate utility classifier produces a 60-dimensional
# public embedding. ConditionalEmbedding transforms that
# embedding before passing it to the U-Net.
# ----------------------------------------------------------

CONDITION_DIM = Z_DIM  # 60

condition_embedding = ConditionalEmbedding(
    num_labels=NUM_ACTIVITIES,
    d_model=Z_DIM,
    dim=CONDITION_DIM
).to(device)


# ----------------------------------------------------------
# U-Net configuration
# ----------------------------------------------------------
# The original U-Net uses GroupNorm(32, channels),
# so channel counts must be divisible by 32.
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    UNET_BASE_CHANNELS = 32
    UNET_CHANNEL_MULTIPLIERS = [1, 2, 4]
    UNET_RESIDUAL_BLOCKS = 1
    UNET_DROPOUT = 0.1

else:

    UNET_BASE_CHANNELS = 64
    UNET_CHANNEL_MULTIPLIERS = [1, 2, 4, 8]
    UNET_RESIDUAL_BLOCKS = 2
    UNET_DROPOUT = 0.1


denoiser_model = Unet(
    in_ch=1,
    mod_ch=UNET_BASE_CHANNELS,
    out_ch=1,
    ch_mul=UNET_CHANNEL_MULTIPLIERS,
    num_res_blocks=UNET_RESIDUAL_BLOCKS,
    cdim=CONDITION_DIM,
    use_conv=True,
    droprate=UNET_DROPOUT,
    dtype=torch.float32
).to(device)


# ----------------------------------------------------------
# IMU padding helper
# ----------------------------------------------------------

def pad_imu_for_unet(x):
    """
    Pad IMU feature width from 30 to 32.

    Input:
        (batch, 1, 128, 30)

    Output:
        (batch, 1, 128, 32)
    """

    target_width = 32
    current_width = x.shape[-1]

    if current_width > target_width:
        raise ValueError(
            f"Input width {current_width} exceeds "
            f"target width {target_width}."
        )

    padding_right = target_width - current_width

    if padding_right > 0:
        x = F.pad(
            x,
            pad=(0, padding_right, 0, 0)
        )

    return x


# ----------------------------------------------------------
# IMU cropping helper
# ----------------------------------------------------------

def crop_imu_from_unet(x):
    """
    Crop U-Net output from 32 features
    back to the original 30 IMU features.
    """

    return x[..., :NUM_FEATURES]


# ----------------------------------------------------------
# Dry-run test
# ----------------------------------------------------------

denoiser_model.eval()
condition_embedding.eval()
surrogate_model.eval()

with torch.no_grad():

    dry_windows = sample_windows.to(device)

    dry_windows_padded = pad_imu_for_unet(
        dry_windows
    )

    dry_timesteps = torch.randint(
        low=0,
        high=DIFFUSION_STEPS,
        size=(dry_windows.shape[0],),
        device=device,
        dtype=torch.long
    )

    # Obtain the 60-dimensional public/activity embedding
    # from the frozen surrogate utility classifier.
    _, dry_z_public = surrogate_model(
        dry_windows
    )

    # Transform the public embedding into the U-Net condition.
    dry_condition = condition_embedding(
        dry_z_public
    )

    dry_prediction_padded = denoiser_model(
        dry_windows_padded,
        dry_timesteps,
        dry_condition
    )

    dry_prediction = crop_imu_from_unet(
        dry_prediction_padded
    )


# ----------------------------------------------------------
# Count trainable parameters
# ----------------------------------------------------------

trainable_parameters = sum(
    parameter.numel()
    for parameter in list(
        denoiser_model.parameters()
    ) + list(
        condition_embedding.parameters()
    )
    if parameter.requires_grad
)


# ----------------------------------------------------------
# Verification
# ----------------------------------------------------------

print("=" * 75)
print("CONDITIONAL U-NET DENOISER CHECK")
print("=" * 75)

print(
    "Original input shape       :",
    tuple(dry_windows.shape)
)

print(
    "Padded input shape         :",
    tuple(dry_windows_padded.shape)
)

print(
    "Public embedding shape     :",
    tuple(dry_z_public.shape)
)

print(
    "Condition embedding shape  :",
    tuple(dry_condition.shape)
)

print(
    "Raw U-Net output shape     :",
    tuple(dry_prediction_padded.shape)
)

print(
    "Cropped prediction shape   :",
    tuple(dry_prediction.shape)
)

print(
    "Trainable parameters       :",
    f"{trainable_parameters:,}"
)

print(
    "\nConditional denoiser initialized successfully."
)


# In[24]:


# ==========================================================
# Cell 9 - Fast Conditional Diffusion Training
# ==========================================================

import time
import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Fast local experiment configuration
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    FAST_EPOCHS = 3
    MAX_BATCHES_PER_EPOCH = 40

else:

    FAST_EPOCHS = 5
    MAX_BATCHES_PER_EPOCH = 500


# ----------------------------------------------------------
# Optimizer
# ----------------------------------------------------------

optimizer = torch.optim.Adam(
    list(denoiser_model.parameters())
    + list(condition_embedding.parameters()),
    lr=LEARNING_RATE
)


# ----------------------------------------------------------
# Training state
# ----------------------------------------------------------

denoiser_model.train()
condition_embedding.train()

surrogate_model.eval()
privacy_model.eval()

training_losses = []

start_time = time.time()


print("=" * 75)
print("FAST CONDITIONAL DIFFUSION TRAINING")
print("=" * 75)

print("Epochs                 :", FAST_EPOCHS)
print("Maximum batches/epoch  :", MAX_BATCHES_PER_EPOCH)
print("Device                 :", device)
print()


# ----------------------------------------------------------
# Training loop
# ----------------------------------------------------------

for epoch in range(FAST_EPOCHS):

    epoch_loss = 0.0
    processed_batches = 0

    for batch_index, batch in enumerate(train_loader):

        if (
            MAX_BATCHES_PER_EPOCH is not None
            and batch_index >= MAX_BATCHES_PER_EPOCH
        ):
            break

        windows = batch[0].to(
            device,
            non_blocking=True
        )

        batch_size = windows.shape[0]

        # Pad from 30 features to 32.
        clean_padded = pad_imu_for_unet(
            windows
        )

        # Random timestep for every sample.
        timesteps = torch.randint(
            low=0,
            high=DIFFUSION_STEPS,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )

        # Sample Gaussian noise.
        true_noise = torch.randn_like(
            clean_padded
        )

        # Add noise using q(x_t | x_0).
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
            + sqrt_one_minus_alpha_bar_t * true_noise
        )

        # Obtain the frozen activity-related representation.
        with torch.no_grad():

            _, z_public = surrogate_model(
                windows
            )

        activity_condition = condition_embedding(
            z_public
        )

        # Predict the Gaussian noise.
        predicted_noise = denoiser_model(
            noisy_padded,
            timesteps,
            activity_condition
        )

        # Standard diffusion noise-prediction loss.
        loss = F.mse_loss(
            predicted_noise,
            true_noise
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(denoiser_model.parameters())
            + list(condition_embedding.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()
        processed_batches += 1

    average_epoch_loss = (
        epoch_loss / max(processed_batches, 1)
    )

    training_losses.append(
        average_epoch_loss
    )

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch + 1}/{FAST_EPOCHS} | "
        f"Batches: {processed_batches} | "
        f"Loss: {average_epoch_loss:.6f} | "
        f"Elapsed: {elapsed:.1f} sec"
    )


# ----------------------------------------------------------
# Save checkpoint
# ----------------------------------------------------------

checkpoint_to_save = {
    "denoiser_state_dict":
        denoiser_model.state_dict(),

    "condition_embedding_state_dict":
        condition_embedding.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "training_losses":
        training_losses,

    "run_mode":
        RUN_MODE,

    "epochs":
        FAST_EPOCHS,

    "diffusion_steps":
        DIFFUSION_STEPS,

    "unet_base_channels":
        UNET_BASE_CHANNELS,

    "unet_channel_multipliers":
        UNET_CHANNEL_MULTIPLIERS,

    "condition_dim":
        CONDITION_DIM
}

torch.save(
    checkpoint_to_save,
    DIFFUSION_CHECKPOINT
)


# ----------------------------------------------------------
# Finish
# ----------------------------------------------------------

total_time = time.time() - start_time

denoiser_model.eval()
condition_embedding.eval()


print("-" * 75)

print(
    "Final training loss :",
    f"{training_losses[-1]:.6f}"
)

print(
    "Total training time :",
    f"{total_time:.1f} seconds"
)

print(
    "Checkpoint saved    :",
    DIFFUSION_CHECKPOINT
)

print(
    "\nFast diffusion training completed successfully."
)


# In[25]:


# ==========================================================
# Cell 10 - Load Trained Diffusion Model and Define Sampling
# ==========================================================

import torch
import torch.nn.functional as F


# ----------------------------------------------------------
# Load saved diffusion checkpoint
# ----------------------------------------------------------

saved_diffusion_checkpoint = torch.load(
    DIFFUSION_CHECKPOINT,
    map_location=device,
    weights_only=False
)

denoiser_model.load_state_dict(
    saved_diffusion_checkpoint[
        "denoiser_state_dict"
    ]
)

condition_embedding.load_state_dict(
    saved_diffusion_checkpoint[
        "condition_embedding_state_dict"
    ]
)

denoiser_model.eval()
condition_embedding.eval()
surrogate_model.eval()
privacy_model.eval()


# ----------------------------------------------------------
# Reverse diffusion step
# ----------------------------------------------------------

@torch.no_grad()
def reverse_diffusion_step(
    x_t,
    timestep,
    condition
):
    """
    Performs one reverse diffusion step:

        x_t -> x_(t-1)
    """

    batch_size = x_t.shape[0]

    timestep_batch = torch.full(
        size=(batch_size,),
        fill_value=timestep,
        device=device,
        dtype=torch.long
    )

    predicted_noise = denoiser_model(
        x_t,
        timestep_batch,
        condition
    )

    beta_t = extract(
        betas,
        timestep_batch,
        x_t.shape
    )

    sqrt_one_minus_alpha_bar_t = extract(
        sqrt_one_minus_alphas_cumprod,
        timestep_batch,
        x_t.shape
    )

    sqrt_recip_alpha_t = extract(
        sqrt_recip_alphas,
        timestep_batch,
        x_t.shape
    )

    model_mean = sqrt_recip_alpha_t * (
        x_t
        - beta_t
        * predicted_noise
        / sqrt_one_minus_alpha_bar_t
    )

    if timestep == 0:
        return model_mean

    posterior_variance_t = extract(
        posterior_variance,
        timestep_batch,
        x_t.shape
    )

    random_noise = torch.randn_like(
        x_t
    )

    return (
        model_mean
        + torch.sqrt(
            posterior_variance_t
        )
        * random_noise
    )


# ----------------------------------------------------------
# Obfuscate one batch
# ----------------------------------------------------------

@torch.no_grad()
def obfuscate_batch(
    clean_windows,
    start_timestep
):
    """
    1. Adds noise up to start_timestep.
    2. Runs reverse diffusion back to timestep zero.
    3. Uses the frozen activity embedding as condition.
    """

    clean_windows = clean_windows.to(
        device
    )

    clean_padded = pad_imu_for_unet(
        clean_windows
    )

    batch_size = clean_windows.shape[0]

    timestep_batch = torch.full(
        size=(batch_size,),
        fill_value=start_timestep,
        device=device,
        dtype=torch.long
    )

    initial_noise = torch.randn_like(
        clean_padded
    )

    noisy_windows, _ = forward_diffusion_sample(
        clean_padded,
        timestep_batch,
        noise=initial_noise
    )

    _, z_public = surrogate_model(
        clean_windows
    )

    condition = condition_embedding(
        z_public
    )

    reconstructed = noisy_windows

    for timestep in reversed(
        range(start_timestep + 1)
    ):
        reconstructed = reverse_diffusion_step(
            reconstructed,
            timestep,
            condition
        )

    reconstructed = crop_imu_from_unet(
        reconstructed
    )

    return reconstructed


# ----------------------------------------------------------
# Quick sampling test
# ----------------------------------------------------------

TEST_TIMESTEP = 5

with torch.no_grad():

    quick_obfuscated_batch = obfuscate_batch(
        sample_windows,
        start_timestep=TEST_TIMESTEP
    )


print("=" * 75)
print("REVERSE DIFFUSION CHECK")
print("=" * 75)

print(
    "Checkpoint loaded       :",
    DIFFUSION_CHECKPOINT.name
)

print(
    "Test timestep           :",
    TEST_TIMESTEP
)

print(
    "Original batch shape    :",
    tuple(sample_windows.shape)
)

print(
    "Obfuscated batch shape  :",
    tuple(quick_obfuscated_batch.shape)
)

print(
    "Original mean           :",
    float(sample_windows.mean())
)

print(
    "Obfuscated mean         :",
    float(
        quick_obfuscated_batch.mean().cpu()
    )
)

print(
    "Mean absolute change    :",
    float(
        torch.mean(
            torch.abs(
                quick_obfuscated_batch.cpu()
                - sample_windows
            )
        )
    )
)

print(
    "\nReverse diffusion sampling completed successfully."
)


# In[26]:


# ==========================================================
# Cell 11 - Fast Privacy–Utility Evaluation Across Timesteps
# CPU and Narval/GPU Compatible
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
# Evaluation configuration
# ----------------------------------------------------------

# Start with three useful privacy/utility operating points.
# Higher timesteps can be added later if time permits.
EVALUATION_TIMESTEPS = [
    5,
    10,
    20
]

if RUN_MODE == "small_cpu":

    MAX_EVALUATION_BATCHES = 10

else:

    # Narval deadline-aware experiment.
    #
    # 4 batches × batch size 64
    # ≈ 256 test windows.
    #
    # This is intentionally capped so that reverse diffusion
    # evaluation completes in practical time.
    MAX_EVALUATION_BATCHES = 4


# ----------------------------------------------------------
# Evaluate raw or obfuscated windows
# ----------------------------------------------------------

@torch.no_grad()
def evaluate_windows(
    data_loader,
    obfuscation_timestep=None,
    max_batches=None
):

    true_activities = []
    predicted_activities = []

    true_genders = []
    predicted_genders = []

    total_absolute_change = 0.0
    total_squared_change = 0.0
    total_values = 0
    total_samples = 0

    surrogate_model.eval()
    privacy_model.eval()
    denoiser_model.eval()
    condition_embedding.eval()

    for batch_index, batch in enumerate(
        data_loader
    ):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        windows, activities, genders, _ = batch

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

        genders = genders.to(
            device,
            dtype=torch.long,
            non_blocking=True
        )

        # --------------------------------------------------
        # Raw or diffusion-obfuscated input
        # --------------------------------------------------

        if obfuscation_timestep is None:

            evaluated_windows = windows

        else:

            evaluated_windows = obfuscate_batch(
                clean_windows=windows,
                start_timestep=obfuscation_timestep
            )

            differences = (
                evaluated_windows - windows
            )

            total_absolute_change += (
                torch.sum(
                    torch.abs(differences)
                ).item()
            )

            total_squared_change += (
                torch.sum(
                    differences ** 2
                ).item()
            )

            total_values += windows.numel()

        # --------------------------------------------------
        # Utility classifier
        # --------------------------------------------------

        activity_logits, _ = surrogate_model(
            evaluated_windows
        )

        activity_predictions = (
            activity_logits.argmax(dim=1)
        )

        # --------------------------------------------------
        # Privacy attacker
        # --------------------------------------------------

        gender_logits, _ = privacy_model(
            evaluated_windows
        )

        gender_predictions = (
            gender_logits.argmax(dim=1)
        )

        # --------------------------------------------------
        # Collect labels
        # --------------------------------------------------

        true_activities.extend(
            activities.cpu().tolist()
        )

        predicted_activities.extend(
            activity_predictions.cpu().tolist()
        )

        true_genders.extend(
            genders.cpu().tolist()
        )

        predicted_genders.extend(
            gender_predictions.cpu().tolist()
        )

        total_samples += windows.shape[0]

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    activity_accuracy = accuracy_score(
        true_activities,
        predicted_activities
    )

    activity_f1 = f1_score(
        true_activities,
        predicted_activities,
        average="macro",
        zero_division=0
    )

    gender_accuracy = accuracy_score(
        true_genders,
        predicted_genders
    )

    gender_f1 = f1_score(
        true_genders,
        predicted_genders,
        average="macro",
        zero_division=0
    )

    if obfuscation_timestep is None:

        mean_absolute_change = 0.0
        root_mean_squared_change = 0.0

    else:

        mean_absolute_change = (
            total_absolute_change
            / max(total_values, 1)
        )

        root_mean_squared_change = np.sqrt(
            total_squared_change
            / max(total_values, 1)
        )

    return {

        "Timestep": (
            "Raw"
            if obfuscation_timestep is None
            else obfuscation_timestep
        ),

        "Samples": total_samples,

        "Activity Accuracy":
            activity_accuracy,

        "Activity Macro F1":
            activity_f1,

        "Gender Accuracy":
            gender_accuracy,

        "Gender Macro F1":
            gender_f1,

        "Mean Absolute Change":
            mean_absolute_change,

        "Root Mean Squared Change":
            root_mean_squared_change
    }


# ==========================================================
# Run Evaluation
# ==========================================================

evaluation_results = []

evaluation_start_time = time.time()

print("=" * 95)
print("FAST PRIVACY–UTILITY EVALUATION")
print("=" * 95)

print(
    "Evaluation timesteps :",
    EVALUATION_TIMESTEPS
)

print(
    "Maximum batches      :",
    MAX_EVALUATION_BATCHES
)

print(
    "Batch size           :",
    BATCH_SIZE
)


# ----------------------------------------------------------
# Raw baseline
# ----------------------------------------------------------

print(
    "\nEvaluating raw test windows..."
)

raw_result = evaluate_windows(
    data_loader=test_loader,
    obfuscation_timestep=None,
    max_batches=MAX_EVALUATION_BATCHES
)

evaluation_results.append(
    raw_result
)

print(
    "Raw complete | "
    f"Samples: {raw_result['Samples']} | "
    f"Activity accuracy: "
    f"{raw_result['Activity Accuracy']:.4f} | "
    f"Activity F1: "
    f"{raw_result['Activity Macro F1']:.4f} | "
    f"Gender accuracy: "
    f"{raw_result['Gender Accuracy']:.4f} | "
    f"Gender F1: "
    f"{raw_result['Gender Macro F1']:.4f}"
)


# ----------------------------------------------------------
# Obfuscated evaluations
# ----------------------------------------------------------

for timestep in EVALUATION_TIMESTEPS:

    timestep_start = time.time()

    print(
        f"\nEvaluating timestep {timestep}..."
    )

    timestep_result = evaluate_windows(
        data_loader=test_loader,
        obfuscation_timestep=timestep,
        max_batches=MAX_EVALUATION_BATCHES
    )

    evaluation_results.append(
        timestep_result
    )

    timestep_elapsed = (
        time.time()
        - timestep_start
    )

    print(
        f"Timestep {timestep} complete | "
        f"Samples: {timestep_result['Samples']} | "
        f"Activity accuracy: "
        f"{timestep_result['Activity Accuracy']:.4f} | "
        f"Activity F1: "
        f"{timestep_result['Activity Macro F1']:.4f} | "
        f"Gender accuracy: "
        f"{timestep_result['Gender Accuracy']:.4f} | "
        f"Gender F1: "
        f"{timestep_result['Gender Macro F1']:.4f} | "
        f"MAC: "
        f"{timestep_result['Mean Absolute Change']:.4f} | "
        f"RMSE: "
        f"{timestep_result['Root Mean Squared Change']:.4f} | "
        f"Time: {timestep_elapsed:.1f} sec"
    )


# ==========================================================
# Create Results Table
# ==========================================================

results_df = pd.DataFrame(
    evaluation_results
)

numeric_columns = [

    "Activity Accuracy",
    "Activity Macro F1",

    "Gender Accuracy",
    "Gender Macro F1",

    "Mean Absolute Change",
    "Root Mean Squared Change"
]

results_df[numeric_columns] = (
    results_df[numeric_columns]
    .round(4)
)


# ----------------------------------------------------------
# Changes relative to raw baseline
# ----------------------------------------------------------

raw_activity_accuracy = float(
    evaluation_results[0][
        "Activity Accuracy"
    ]
)

raw_gender_accuracy = float(
    evaluation_results[0][
        "Gender Accuracy"
    ]
)

results_df[
    "Activity Accuracy Change"
] = (
    results_df[
        "Activity Accuracy"
    ]
    - raw_activity_accuracy
).round(4)

results_df[
    "Gender Accuracy Change"
] = (
    results_df[
        "Gender Accuracy"
    ]
    - raw_gender_accuracy
).round(4)


# ==========================================================
# Save Results
# ==========================================================

if RUN_MODE == "full_gpu":

    RESULTS_CSV = (
        WORK_DIR
        / "privacy_utility_results.csv"
    )

else:

    RESULTS_CSV = (
        WORK_DIR
        / "small_cpu_privacy_utility_results.csv"
    )

results_df.to_csv(
    RESULTS_CSV,
    index=False
)


# ==========================================================
# Final Output
# ==========================================================

total_evaluation_time = (
    time.time()
    - evaluation_start_time
)

print(
    "\n" + "=" * 110
)

print(
    "PRIVACY–UTILITY RESULTS"
)

print(
    "=" * 110
)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "-" * 110
)

print(
    "Total evaluation time:",
    f"{total_evaluation_time:.1f} seconds"
)

print(
    "Results saved to:",
    RESULTS_CSV
)

print(
    "\nEvaluation completed successfully."
)


# In[27]:


# ==========================================================
# Cell 12 - Balanced Privacy–Utility Evaluation
# ==========================================================

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)


# ----------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------

EVALUATION_TIMESTEPS = [
    5,
    10,
    20,
    30,
    40
]

# Test set contains four subjects from the smaller gender group.
# Five windows × six activities × four subjects × two genders
# gives a maximum of 240 balanced evaluation windows.
WINDOWS_PER_ACTIVITY_PER_SUBJECT = 5

# Set to None to evaluate the complete balanced subset.
MAX_EVALUATION_BATCHES = None


# ----------------------------------------------------------
# Balanced evaluation dataset
# ----------------------------------------------------------

class BalancedEvaluationDataset(Dataset):
    """
    Creates a test subset balanced across:

        1. Gender
        2. Activity
        3. Number of selected subjects
        4. Windows contributed by each subject

    Each item returns:

        window   : (1, 128, 30), float32
        activity : scalar long
        gender   : scalar long
        weight   : scalar long
    """

    def __init__(
        self,
        processed_dir,
        windows_per_activity_per_subject=5
    ):
        super().__init__()

        self.processed_dir = Path(
            processed_dir
        )

        self.files = sorted(
            self.processed_dir.glob(
                "*_test.npz"
            )
        )

        if not self.files:
            raise FileNotFoundError(
                "No test .npz files were found in:\n"
                f"{self.processed_dir}"
            )

        gender_to_files = {
            0: [],
            1: []
        }

        # Group test subjects by gender.
        for file_path in self.files:

            with np.load(file_path) as data:

                gender = int(
                    data["gender"][0]
                )

            if gender not in gender_to_files:
                raise ValueError(
                    f"Unexpected gender label {gender} "
                    f"in {file_path.name}"
                )

            gender_to_files[gender].append(
                file_path
            )

        available_gender_0 = len(
            gender_to_files[0]
        )

        available_gender_1 = len(
            gender_to_files[1]
        )

        if (
            available_gender_0 == 0
            or available_gender_1 == 0
        ):
            raise ValueError(
                "Both gender classes must be represented "
                "in the test data."
            )

        # Use the same number of subjects from each gender.
        subjects_per_gender = min(
            available_gender_0,
            available_gender_1
        )

        self.selected_subjects = {
            0: gender_to_files[0][
                :subjects_per_gender
            ],
            1: gender_to_files[1][
                :subjects_per_gender
            ]
        }

        self.samples = []

        # Select an equal number of windows for every activity
        # from every selected subject.
        for gender in [0, 1]:

            for file_path in self.selected_subjects[
                gender
            ]:

                with np.load(file_path) as data:

                    activity_labels = np.asarray(
                        data["activity"],
                        dtype=np.int64
                    )

                for activity_class in range(
                    NUM_ACTIVITIES
                ):

                    class_indices = np.where(
                        activity_labels
                        == activity_class
                    )[0]

                    if len(class_indices) == 0:
                        raise ValueError(
                            f"{file_path.name} has no windows "
                            f"for activity {activity_class}."
                        )

                    number_to_select = min(
                        windows_per_activity_per_subject,
                        len(class_indices)
                    )

                    # Select windows from across the entire
                    # activity segment.
                    positions = np.linspace(
                        0,
                        len(class_indices) - 1,
                        num=number_to_select,
                        dtype=np.int64
                    )

                    selected_indices = (
                        class_indices[positions]
                    )

                    for local_index in selected_indices:

                        self.samples.append(
                            (
                                file_path,
                                int(local_index)
                            )
                        )

        self.available_subject_counts = {
            0: available_gender_0,
            1: available_gender_1
        }

        self.subjects_per_gender = (
            subjects_per_gender
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        file_path, local_index = (
            self.samples[index]
        )

        with np.load(file_path) as data:

            window = np.asarray(
                data["windows"][local_index],
                dtype=np.float32
            ).copy()

            activity = int(
                data["activity"][local_index]
            )

            gender = int(
                data["gender"][0]
            )

            weight = int(
                data["weight"][0]
            )

        window = torch.from_numpy(
            window
        ).unsqueeze(0)

        return (
            window,
            torch.tensor(
                activity,
                dtype=torch.long
            ),
            torch.tensor(
                gender,
                dtype=torch.long
            ),
            torch.tensor(
                weight,
                dtype=torch.long
            )
        )


# ----------------------------------------------------------
# Create balanced evaluation DataLoader
# ----------------------------------------------------------

balanced_test_dataset = BalancedEvaluationDataset(
    processed_dir=PROCESSED_DIR,
    windows_per_activity_per_subject=(
        WINDOWS_PER_ACTIVITY_PER_SUBJECT
    )
)

balanced_test_loader = DataLoader(
    balanced_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        device.type == "cuda"
    ),
    drop_last=False
)


# ----------------------------------------------------------
# Verify complete balanced distribution
# ----------------------------------------------------------

activity_counts = torch.zeros(
    NUM_ACTIVITIES,
    dtype=torch.long
)

gender_counts = torch.zeros(
    NUM_PRIVATE_CLASSES,
    dtype=torch.long
)

joint_counts = torch.zeros(
    (
        NUM_PRIVATE_CLASSES,
        NUM_ACTIVITIES
    ),
    dtype=torch.long
)

for _, activities, genders, _ in (
    balanced_test_loader
):

    activity_counts += torch.bincount(
        activities,
        minlength=NUM_ACTIVITIES
    )

    gender_counts += torch.bincount(
        genders,
        minlength=NUM_PRIVATE_CLASSES
    )

    for gender, activity in zip(
        genders,
        activities
    ):

        joint_counts[
            int(gender),
            int(activity)
        ] += 1


print("=" * 90)
print("BALANCED EVALUATION DATASET")
print("=" * 90)

print(
    "Available test subjects by gender :",
    balanced_test_dataset.available_subject_counts
)

print(
    "Selected subjects per gender      :",
    balanced_test_dataset.subjects_per_gender
)

print(
    "Total balanced windows            :",
    len(balanced_test_dataset)
)

print(
    "Activity counts                   :",
    activity_counts.tolist()
)

print(
    "Gender counts                     :",
    gender_counts.tolist()
)

print(
    "Gender × activity counts:"
)

print(
    joint_counts.tolist()
)


# Stop immediately if balancing failed.
if not torch.all(
    activity_counts
    == activity_counts[0]
):
    raise RuntimeError(
        "The evaluation activities are not balanced."
    )

if not torch.all(
    gender_counts
    == gender_counts[0]
):
    raise RuntimeError(
        "The evaluation genders are not balanced."
    )

if not torch.all(
    joint_counts
    == joint_counts[0, 0]
):
    raise RuntimeError(
        "The joint gender-activity distribution "
        "is not balanced."
    )

print(
    "\nBalanced evaluation distribution verified."
)


# ----------------------------------------------------------
# Evaluation function
# ----------------------------------------------------------

@torch.no_grad()
def evaluate_windows(
    data_loader,
    obfuscation_timestep=None,
    max_batches=None
):
    """
    Evaluates activity utility and gender inference.

    obfuscation_timestep=None:
        Evaluate the original test windows.

    obfuscation_timestep=int:
        Obfuscate the windows through forward and reverse
        diffusion before evaluation.
    """

    true_activities = []
    predicted_activities = []

    true_genders = []
    predicted_genders = []

    total_absolute_change = 0.0
    total_squared_change = 0.0
    total_values = 0
    total_samples = 0

    surrogate_model.eval()
    privacy_model.eval()
    denoiser_model.eval()
    condition_embedding.eval()

    for batch_index, batch in enumerate(
        data_loader
    ):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

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

        if obfuscation_timestep is None:

            evaluated_windows = windows

        else:

            evaluated_windows = obfuscate_batch(
                clean_windows=windows,
                start_timestep=(
                    obfuscation_timestep
                )
            )

            difference = (
                evaluated_windows - windows
            )

            total_absolute_change += (
                torch.sum(
                    torch.abs(difference)
                ).item()
            )

            total_squared_change += (
                torch.sum(
                    difference ** 2
                ).item()
            )

            total_values += windows.numel()

        activity_logits, _ = surrogate_model(
            evaluated_windows
        )

        gender_logits, _ = privacy_model(
            evaluated_windows
        )

        activity_predictions = (
            activity_logits.argmax(dim=1)
        )

        gender_predictions = (
            gender_logits.argmax(dim=1)
        )

        true_activities.extend(
            activities.cpu().tolist()
        )

        predicted_activities.extend(
            activity_predictions.cpu().tolist()
        )

        true_genders.extend(
            genders.cpu().tolist()
        )

        predicted_genders.extend(
            gender_predictions.cpu().tolist()
        )

        total_samples += windows.shape[0]

    if total_samples == 0:
        raise RuntimeError(
            "No evaluation samples were processed."
        )

    activity_accuracy = accuracy_score(
        true_activities,
        predicted_activities
    )

    activity_macro_f1 = f1_score(
        true_activities,
        predicted_activities,
        average="macro",
        zero_division=0
    )

    gender_accuracy = accuracy_score(
        true_genders,
        predicted_genders
    )

    gender_macro_f1 = f1_score(
        true_genders,
        predicted_genders,
        average="macro",
        zero_division=0
    )

    activity_confusion = confusion_matrix(
        true_activities,
        predicted_activities,
        labels=list(
            range(NUM_ACTIVITIES)
        )
    )

    gender_confusion = confusion_matrix(
        true_genders,
        predicted_genders,
        labels=list(
            range(NUM_PRIVATE_CLASSES)
        )
    )

    if obfuscation_timestep is None:

        mean_absolute_change = 0.0
        mean_squared_change = 0.0
        root_mean_squared_change = 0.0

    else:

        mean_absolute_change = (
            total_absolute_change
            / max(total_values, 1)
        )

        mean_squared_change = (
            total_squared_change
            / max(total_values, 1)
        )

        root_mean_squared_change = (
            mean_squared_change ** 0.5
        )

    return {
        "Timestep": (
            "Raw"
            if obfuscation_timestep is None
            else obfuscation_timestep
        ),
        "Samples": total_samples,
        "Activity Accuracy":
            activity_accuracy,
        "Activity Macro F1":
            activity_macro_f1,
        "Gender Accuracy":
            gender_accuracy,
        "Gender Macro F1":
            gender_macro_f1,
        "Mean Absolute Change":
            mean_absolute_change,
        "Mean Squared Change":
            mean_squared_change,
        "Root Mean Squared Change":
            root_mean_squared_change,
        "Activity Confusion Matrix":
            activity_confusion,
        "Gender Confusion Matrix":
            gender_confusion
    }


# ----------------------------------------------------------
# Run raw and obfuscated evaluations
# ----------------------------------------------------------

evaluation_results = []
detailed_results = {}

evaluation_start_time = time.time()

print("\n" + "=" * 90)
print("BALANCED PRIVACY–UTILITY EVALUATION")
print("=" * 90)

print(
    "Evaluation timesteps:",
    EVALUATION_TIMESTEPS
)

print(
    "Evaluation samples  :",
    len(balanced_test_dataset)
)

print(
    "Maximum batches     :",
    MAX_EVALUATION_BATCHES
)


# Raw baseline
print(
    "\nEvaluating raw balanced test windows..."
)

raw_result = evaluate_windows(
    data_loader=balanced_test_loader,
    obfuscation_timestep=None,
    max_batches=MAX_EVALUATION_BATCHES
)

evaluation_results.append(
    {
        key: value
        for key, value in raw_result.items()
        if "Confusion Matrix" not in key
    }
)

detailed_results["Raw"] = raw_result

print(
    "Raw complete | "
    f"Activity accuracy: "
    f"{raw_result['Activity Accuracy']:.4f} | "
    f"Activity F1: "
    f"{raw_result['Activity Macro F1']:.4f} | "
    f"Gender accuracy: "
    f"{raw_result['Gender Accuracy']:.4f} | "
    f"Gender F1: "
    f"{raw_result['Gender Macro F1']:.4f}"
)


# Diffusion timesteps
for timestep in EVALUATION_TIMESTEPS:

    timestep_start = time.time()

    print(
        f"\nEvaluating timestep {timestep}..."
    )

    timestep_result = evaluate_windows(
        data_loader=balanced_test_loader,
        obfuscation_timestep=timestep,
        max_batches=MAX_EVALUATION_BATCHES
    )

    evaluation_results.append(
        {
            key: value
            for key, value
            in timestep_result.items()
            if "Confusion Matrix" not in key
        }
    )

    detailed_results[
        timestep
    ] = timestep_result

    timestep_elapsed = (
        time.time() - timestep_start
    )

    print(
        f"Timestep {timestep} complete | "
        f"Activity accuracy: "
        f"{timestep_result['Activity Accuracy']:.4f} | "
        f"Activity F1: "
        f"{timestep_result['Activity Macro F1']:.4f} | "
        f"Gender accuracy: "
        f"{timestep_result['Gender Accuracy']:.4f} | "
        f"Gender F1: "
        f"{timestep_result['Gender Macro F1']:.4f} | "
        f"MAC: "
        f"{timestep_result['Mean Absolute Change']:.4f} | "
        f"Time: {timestep_elapsed:.1f} sec"
    )


# ----------------------------------------------------------
# Create results DataFrame
# ----------------------------------------------------------

results_df = pd.DataFrame(
    evaluation_results
)

numeric_columns = [
    "Activity Accuracy",
    "Activity Macro F1",
    "Gender Accuracy",
    "Gender Macro F1",
    "Mean Absolute Change",
    "Mean Squared Change",
    "Root Mean Squared Change"
]

results_df[numeric_columns] = (
    results_df[numeric_columns].round(4)
)


# ----------------------------------------------------------
# Calculate change relative to raw baseline
# ----------------------------------------------------------

raw_activity_accuracy = float(
    raw_result["Activity Accuracy"]
)

raw_gender_accuracy = float(
    raw_result["Gender Accuracy"]
)

results_df[
    "Activity Accuracy Change"
] = (
    results_df["Activity Accuracy"]
    - raw_activity_accuracy
).round(4)

results_df[
    "Gender Accuracy Change"
] = (
    results_df["Gender Accuracy"]
    - raw_gender_accuracy
).round(4)


# ----------------------------------------------------------
# Save results
# ----------------------------------------------------------

RESULTS_CSV = (
    WORK_DIR
    / "balanced_privacy_utility_results.csv"
)

RESULTS_PT = (
    WORK_DIR
    / "balanced_privacy_utility_details.pt"
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)

torch.save(
    {
        "results_table":
            results_df.to_dict(
                orient="records"
            ),
        "activity_confusion_matrices":
            {
                str(key):
                    value[
                        "Activity Confusion Matrix"
                    ]
                for key, value
                in detailed_results.items()
            },
        "gender_confusion_matrices":
            {
                str(key):
                    value[
                        "Gender Confusion Matrix"
                    ]
                for key, value
                in detailed_results.items()
            },
        "activity_counts":
            activity_counts.tolist(),
        "gender_counts":
            gender_counts.tolist(),
        "joint_counts":
            joint_counts.tolist(),
        "timesteps":
            EVALUATION_TIMESTEPS
    },
    RESULTS_PT
)


# ----------------------------------------------------------
# Display final table
# ----------------------------------------------------------

total_evaluation_time = (
    time.time() - evaluation_start_time
)

display_columns = [
    "Timestep",
    "Samples",
    "Activity Accuracy",
    "Activity Macro F1",
    "Gender Accuracy",
    "Gender Macro F1",
    "Mean Absolute Change",
    "Root Mean Squared Change",
    "Activity Accuracy Change",
    "Gender Accuracy Change"
]


print("\n" + "=" * 110)
print("BALANCED PRIVACY–UTILITY RESULTS")
print("=" * 110)

print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)

print("-" * 110)

print(
    "Total evaluation time:",
    f"{total_evaluation_time:.1f} seconds"
)

print(
    "Results CSV saved to:",
    RESULTS_CSV
)

print(
    "Detailed results saved to:",
    RESULTS_PT
)

print(
    "\nBalanced privacy–utility evaluation completed successfully."
)

print(
    "Results and detailed evaluation metrics have been saved "
    "for further analysis and comparison."
)


# In[ ]:




