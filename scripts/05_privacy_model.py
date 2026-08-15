#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================================
# Cell 1 - Privacy Classifier Configuration
# ==========================================================

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


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
#     Uses a small balanced subset for quick local testing.
#
# full_gpu:
#     Uses the complete multi-subject dataset on Narval.
# ----------------------------------------------------------

RUN_MODE = os.environ.get(
    "RUN_MODE",
    "small_cpu"
)

if RUN_MODE not in {
    "small_cpu",
    "full_gpu"
}:
    raise ValueError(
        "RUN_MODE must be either "
        "'small_cpu' or 'full_gpu'."
    )


# ----------------------------------------------------------
# Processed dataset directory
# ----------------------------------------------------------

PROCESSED_DIR = Path(
    os.environ.get(
        "IMU_PROCESSED_PATH",
        "../datasets/"
        "DatasetIMUandBIOMARKERS/"
        "processed"
    )
).expanduser().resolve()

if not PROCESSED_DIR.exists():
    raise FileNotFoundError(
        "Processed dataset directory was not found:\n"
        f"{PROCESSED_DIR}"
    )


# ----------------------------------------------------------
# Select the private attribute
# ----------------------------------------------------------
# Available choices:
#     "gender" -> binary classification, 2 classes
#     "weight" -> three-class classification
# ----------------------------------------------------------

PRIVATE_ATTRIBUTE = os.environ.get(
    "PRIVATE_ATTRIBUTE",
    "gender"
).lower()

if PRIVATE_ATTRIBUTE == "gender":
    NUM_PRIVATE_CLASSES = 2

elif PRIVATE_ATTRIBUTE == "weight":
    NUM_PRIVATE_CLASSES = 3

else:
    raise ValueError(
        "PRIVATE_ATTRIBUTE must be either "
        "'gender' or 'weight'."
    )


# ----------------------------------------------------------
# Model and training configuration
# ----------------------------------------------------------

Z_DIM = 60

if RUN_MODE == "small_cpu":

    BATCH_SIZE = 8
    EPOCHS = 5
    LEARNING_RATE = 0.001

else:

    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001


# ----------------------------------------------------------
# Configuration summary
# ----------------------------------------------------------

print("=" * 70)
print("PRIVACY CLASSIFIER CONFIGURATION")
print("=" * 70)

print("Run mode              :", RUN_MODE)
print("Device                :", device)
print("Private attribute     :", PRIVATE_ATTRIBUTE)
print("Private classes       :", NUM_PRIVATE_CLASSES)
print("Embedding dimension   :", Z_DIM)
print("Batch size            :", BATCH_SIZE)
print("Epochs                :", EPOCHS)
print("Learning rate         :", LEARNING_RATE)
print("Processed directory   :", PROCESSED_DIR)


# In[2]:


# ==========================================================
# Cell 2 - Balanced Dataset for Gender Privacy Classifier
# ==========================================================

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


PROCESSED_DIR = Path(PROCESSED_DIR)

NUM_ACTIVITIES = 6
NUM_GENDERS = 2
NUM_WEIGHT_CLASSES = 3


def find_subjects_by_gender(processed_directory, split):
    """
    Find available subject files and group them according
    to their gender label.
    """

    files_by_gender = {
        0: [],
        1: []
    }

    subject_files = sorted(
        processed_directory.glob(f"*_{split}.npz")
    )

    for file_path in subject_files:

        with np.load(file_path, allow_pickle=False) as data:
            gender_value = int(
                np.asarray(data["gender"]).reshape(-1)[0]
            )

        if gender_value in files_by_gender:
            files_by_gender[gender_value].append(file_path)

    return files_by_gender


train_files_by_gender = find_subjects_by_gender(
    PROCESSED_DIR,
    split="train"
)

test_files_by_gender = find_subjects_by_gender(
    PROCESSED_DIR,
    split="test"
)


print("Available training subjects:")
print("Gender 0:", len(train_files_by_gender[0]))
print("Gender 1:", len(train_files_by_gender[1]))

print("\nAvailable testing subjects:")
print("Gender 0:", len(test_files_by_gender[0]))
print("Gender 1:", len(test_files_by_gender[1]))


# ----------------------------------------------------------
# Use all balanced training subjects
# ----------------------------------------------------------

subjects_per_gender = min(
    len(train_files_by_gender[0]),
    len(train_files_by_gender[1])
)

selected_train_files = (
    train_files_by_gender[0][:subjects_per_gender]
    + train_files_by_gender[1][:subjects_per_gender]
)

# ----------------------------------------------------------
# Use all balanced testing subjects
# ----------------------------------------------------------

test_subjects_per_gender = min(
    len(test_files_by_gender[0]),
    len(test_files_by_gender[1])
)

selected_test_files = (
    test_files_by_gender[0][:test_subjects_per_gender]
    + test_files_by_gender[1][:test_subjects_per_gender]
)


print("\nSelected training subjects:")
for file_path in selected_train_files:
    print(file_path.name)

print("\nSelected testing subjects:")
for file_path in selected_test_files:
    print(file_path.name)


def create_privacy_dataset(
    subject_files,
    samples_per_activity=20,
    seed=42
):
    """
    Create a small multi-subject dataset for gender prediction.

    An equal number of windows is selected from each activity
    for every chosen participant.
    """

    rng = np.random.default_rng(seed)

    all_windows = []
    all_activities = []
    all_genders = []
    all_weights = []

    for file_path in subject_files:

        with np.load(file_path, allow_pickle=False) as data:

            activity_labels = np.asarray(
                data["activity"],
                dtype=np.int64
            )

            gender_value = int(
                np.asarray(data["gender"]).reshape(-1)[0]
            )

            weight_value = int(
                np.asarray(data["weight"]).reshape(-1)[0]
            )

            selected_indices = []

            for activity_id in range(NUM_ACTIVITIES):

                activity_indices = np.flatnonzero(
                    activity_labels == activity_id
                )

                if len(activity_indices) == 0:
                    continue

                number_to_select = min(
                    samples_per_activity,
                    len(activity_indices)
                )

                chosen_indices = rng.choice(
                    activity_indices,
                    size=number_to_select,
                    replace=False
                )

                selected_indices.extend(
                    chosen_indices.tolist()
                )

            selected_indices = np.asarray(
                selected_indices,
                dtype=np.int64
            )

            rng.shuffle(selected_indices)

            # Decompress the large windows array only once
            subject_windows = data["windows"]

            selected_windows = np.asarray(
                subject_windows[selected_indices],
                dtype=np.float32
            ).copy()

            selected_activities = activity_labels[
                selected_indices
            ].copy()

        number_of_selected_windows = len(selected_indices)

        all_windows.append(selected_windows)
        all_activities.append(selected_activities)

        all_genders.append(
            np.full(
                number_of_selected_windows,
                gender_value,
                dtype=np.int64
            )
        )

        all_weights.append(
            np.full(
                number_of_selected_windows,
                weight_value,
                dtype=np.int64
            )
        )

        print(
            f"Loaded {file_path.name}: "
            f"{number_of_selected_windows} windows, "
            f"gender={gender_value}"
        )

        del subject_windows
        del selected_windows
        gc.collect()

    windows_array = np.concatenate(
        all_windows,
        axis=0
    )

    activity_array = np.concatenate(
        all_activities,
        axis=0
    )

    gender_array = np.concatenate(
        all_genders,
        axis=0
    )

    weight_array = np.concatenate(
        all_weights,
        axis=0
    )

    # Shuffle the complete multi-subject dataset
    shuffle_indices = rng.permutation(
        len(windows_array)
    )

    windows_array = windows_array[shuffle_indices]
    activity_array = activity_array[shuffle_indices]
    gender_array = gender_array[shuffle_indices]
    weight_array = weight_array[shuffle_indices]

    windows_tensor = torch.from_numpy(
        windows_array
    ).unsqueeze(1)

    activity_tensor = F.one_hot(
        torch.from_numpy(activity_array),
        num_classes=NUM_ACTIVITIES
    ).float()

    gender_tensor = F.one_hot(
        torch.from_numpy(gender_array),
        num_classes=NUM_GENDERS
    ).float()

    weight_tensor = F.one_hot(
        torch.from_numpy(weight_array),
        num_classes=NUM_WEIGHT_CLASSES
    ).float()

    print(
        "\nGender counts:",
        torch.bincount(
            torch.from_numpy(gender_array),
            minlength=NUM_GENDERS
        ).tolist()
    )

    return TensorDataset(
        windows_tensor,
        activity_tensor,
        gender_tensor,
        weight_tensor
    )


# Four training subjects:
# 4 × 6 activities × 20 windows = approximately 480 samples
privacy_train_dataset = create_privacy_dataset(
    selected_train_files,
    samples_per_activity=20,
    seed=42
)

# Two held-out testing subjects:
# 2 × 6 activities × 20 windows = approximately 240 samples
privacy_test_dataset = create_privacy_dataset(
    selected_test_files,
    samples_per_activity=20,
    seed=43
)


privacy_train_loader = DataLoader(
    privacy_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

privacy_test_loader = DataLoader(
    privacy_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nPrivacy dataset successfully created")
print("Privacy training samples:", len(privacy_train_dataset))
print("Privacy testing samples :", len(privacy_test_dataset))


privacy_batch = next(
    iter(privacy_train_loader)
)

privacy_windows, privacy_activity, privacy_gender, privacy_weight = (
    privacy_batch
)

print("\nOne privacy training batch:")
print("Windows shape :", tuple(privacy_windows.shape))
print("Activity shape:", tuple(privacy_activity.shape))
print("Gender shape  :", tuple(privacy_gender.shape))
print("Weight shape  :", tuple(privacy_weight.shape))


# In[3]:


# ==========================================================
# Balanced Privacy Dataset
# Preload ONLY selected balanced windows into RAM
# ==========================================================

class BalancedPrivacyIMUDataset(Dataset):

    def __init__(
        self,
        processed_dir,
        split="train",
        private_attribute="gender",
        samples_per_activity_per_subject=500,
        seed=42
    ):

        super().__init__()

        self.processed_dir = Path(
            processed_dir
        ).expanduser().resolve()

        self.split = split

        self.private_attribute = (
            private_attribute.lower()
        )

        self.requested_samples = int(
            samples_per_activity_per_subject
        )

        self.seed = seed

        self.rng = np.random.default_rng(
            seed
        )

        # --------------------------------------------------
        # Validate configuration
        # --------------------------------------------------

        if self.split not in {
            "train",
            "test"
        }:
            raise ValueError(
                "split must be either 'train' or 'test'."
            )

        if self.private_attribute not in {
            "gender",
            "weight"
        }:
            raise ValueError(
                "private_attribute must be either "
                "'gender' or 'weight'."
            )

        if not self.processed_dir.exists():
            raise FileNotFoundError(
                "Processed directory was not found:\n"
                f"{self.processed_dir}"
            )

        if self.requested_samples <= 0:
            raise ValueError(
                "samples_per_activity_per_subject "
                "must be greater than zero."
            )

        self.num_private_classes = (
            2
            if self.private_attribute == "gender"
            else 3
        )

        self.num_activities = 6

        # --------------------------------------------------
        # Find subject files
        # --------------------------------------------------

        all_files = sorted(
            self.processed_dir.glob(
                f"*_{self.split}.npz"
            )
        )

        if not all_files:
            raise FileNotFoundError(
                f"No '*_{self.split}.npz' files "
                f"were found in:\n"
                f"{self.processed_dir}"
            )

        # --------------------------------------------------
        # First pass:
        # inspect files and group subjects by private class
        # --------------------------------------------------

        files_by_private_class = {
            class_id: []
            for class_id in range(
                self.num_private_classes
            )
        }

        file_information = {}

        print("\n" + "=" * 75)
        print(
            f"SCANNING {self.split.upper()} "
            f"PRIVACY SUBJECTS"
        )
        print("=" * 75)

        for file_index, file_path in enumerate(
            all_files
        ):

            with np.load(
                file_path,
                allow_pickle=False
            ) as data:

                required_keys = {
                    "windows",
                    "activity",
                    "gender",
                    "weight"
                }

                missing_keys = (
                    required_keys
                    - set(data.files)
                )

                if missing_keys:
                    raise KeyError(
                        f"{file_path.name} is missing "
                        f"{sorted(missing_keys)}"
                    )

                windows_shape = (
                    data["windows"].shape
                )

                if len(windows_shape) != 3:
                    raise ValueError(
                        f"Invalid windows shape in "
                        f"{file_path.name}: "
                        f"{windows_shape}"
                    )

                if windows_shape[1:] != (
                    128,
                    30
                ):
                    raise ValueError(
                        f"Unexpected window shape in "
                        f"{file_path.name}: "
                        f"{windows_shape}"
                    )

                activity_labels = np.asarray(
                    data["activity"],
                    dtype=np.int64
                )

                if (
                    windows_shape[0]
                    != len(activity_labels)
                ):
                    raise ValueError(
                        f"Window/activity mismatch in "
                        f"{file_path.name}"
                    )

                gender_value = int(
                    np.asarray(
                        data["gender"]
                    ).reshape(-1)[0]
                )

                weight_value = int(
                    np.asarray(
                        data["weight"]
                    ).reshape(-1)[0]
                )

            private_value = (
                gender_value
                if self.private_attribute == "gender"
                else weight_value
            )

            if private_value not in (
                files_by_private_class
            ):
                raise ValueError(
                    f"Invalid {self.private_attribute} "
                    f"label {private_value} in "
                    f"{file_path.name}"
                )

            activity_counts = {
                activity_id: int(
                    np.sum(
                        activity_labels
                        == activity_id
                    )
                )
                for activity_id
                in range(
                    self.num_activities
                )
            }

            files_by_private_class[
                private_value
            ].append(
                file_path
            )

            file_information[
                file_path
            ] = {
                "private_value":
                    private_value,

                "gender":
                    gender_value,

                "weight":
                    weight_value,

                "activity_counts":
                    activity_counts
            }

        # --------------------------------------------------
        # Balance number of subjects
        # --------------------------------------------------

        available_subject_counts = {
            class_id: len(class_files)
            for class_id, class_files
            in files_by_private_class.items()
        }

        subjects_per_class = min(
            available_subject_counts.values()
        )

        if subjects_per_class == 0:
            raise ValueError(
                "At least one private class "
                "contains no subjects."
            )

        selected_files = []

        for class_id in range(
            self.num_private_classes
        ):

            class_files = (
                files_by_private_class[
                    class_id
                ]
            )

            selected_positions = (
                self.rng.choice(
                    len(class_files),
                    size=subjects_per_class,
                    replace=False
                )
            )

            for position in selected_positions:

                selected_files.append(
                    class_files[
                        int(position)
                    ]
                )

        self.files = sorted(
            selected_files
        )

        # --------------------------------------------------
        # Determine common samples/activity/subject
        # --------------------------------------------------

        available_activity_counts = []

        for file_path in self.files:

            for activity_id in range(
                self.num_activities
            ):

                count = (
                    file_information[
                        file_path
                    ]["activity_counts"][
                        activity_id
                    ]
                )

                if count == 0:
                    raise ValueError(
                        f"{file_path.name} contains "
                        f"no windows for Activity "
                        f"{activity_id}"
                    )

                available_activity_counts.append(
                    count
                )

        minimum_available_count = min(
            available_activity_counts
        )

        self.samples_per_activity_per_subject = min(
            self.requested_samples,
            minimum_available_count
        )

        # --------------------------------------------------
        # IMPORTANT FIX:
        # Load each selected subject file ONCE.
        # Extract only the balanced windows we need.
        # --------------------------------------------------

        sampled_windows = []
        sampled_activities = []
        sampled_genders = []
        sampled_weights = []

        print("\n" + "=" * 75)
        print(
            f"PRELOADING BALANCED "
            f"{self.split.upper()} PRIVACY DATASET"
        )
        print("=" * 75)

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

                windows = np.asarray(
                    data["windows"],
                    dtype=np.float32
                )

                activity_labels = np.asarray(
                    data["activity"],
                    dtype=np.int64
                )

                gender_value = int(
                    np.asarray(
                        data["gender"]
                    ).reshape(-1)[0]
                )

                weight_value = int(
                    np.asarray(
                        data["weight"]
                    ).reshape(-1)[0]
                )

                for activity_id in range(
                    self.num_activities
                ):

                    activity_indices = (
                        np.flatnonzero(
                            activity_labels
                            == activity_id
                        )
                    )

                    selected_indices = (
                        self.rng.choice(
                            activity_indices,
                            size=(
                                self.samples_per_activity_per_subject
                            ),
                            replace=False
                        )
                    )

                    selected_windows = np.asarray(
                        windows[
                            selected_indices
                        ],
                        dtype=np.float32
                    ).copy()

                    selected_activities = np.full(
                        len(selected_indices),
                        activity_id,
                        dtype=np.int64
                    )

                    selected_genders = np.full(
                        len(selected_indices),
                        gender_value,
                        dtype=np.int64
                    )

                    selected_weights = np.full(
                        len(selected_indices),
                        weight_value,
                        dtype=np.int64
                    )

                    sampled_windows.append(
                        selected_windows
                    )

                    sampled_activities.append(
                        selected_activities
                    )

                    sampled_genders.append(
                        selected_genders
                    )

                    sampled_weights.append(
                        selected_weights
                    )

        # --------------------------------------------------
        # Combine selected windows ONCE
        # --------------------------------------------------

        self.windows = np.concatenate(
            sampled_windows,
            axis=0
        )

        self.activities = np.concatenate(
            sampled_activities,
            axis=0
        )

        self.genders = np.concatenate(
            sampled_genders,
            axis=0
        )

        self.weights = np.concatenate(
            sampled_weights,
            axis=0
        )

        # --------------------------------------------------
        # Shuffle selected balanced samples
        # --------------------------------------------------

        permutation = self.rng.permutation(
            len(self.windows)
        )

        self.windows = (
            self.windows[
                permutation
            ]
        )

        self.activities = (
            self.activities[
                permutation
            ]
        )

        self.genders = (
            self.genders[
                permutation
            ]
        )

        self.weights = (
            self.weights[
                permutation
            ]
        )

        # --------------------------------------------------
        # Subject count summary
        # --------------------------------------------------

        self.selected_subject_counts = {
            class_id: 0
            for class_id in range(
                self.num_private_classes
            )
        }

        for file_path in self.files:

            class_id = (
                file_information[
                    file_path
                ]["private_value"]
            )

            self.selected_subject_counts[
                class_id
            ] += 1

        # --------------------------------------------------
        # Dataset summary
        # --------------------------------------------------

        print("\n" + "=" * 75)
        print(
            f"BALANCED {self.split.upper()} "
            f"PRIVACY DATASET READY"
        )
        print("=" * 75)

        print(
            "Private attribute          :",
            self.private_attribute
        )

        print(
            "Available subject counts   :",
            available_subject_counts
        )

        print(
            "Selected subject counts    :",
            self.selected_subject_counts
        )

        print(
            "Selected subject files     :",
            len(self.files)
        )

        print(
            "Activities                 :",
            self.num_activities
        )

        print(
            "Requested windows/activity :",
            self.requested_samples
        )

        print(
            "Used windows/activity/"
            "subject:",
            self.samples_per_activity_per_subject
        )

        print(
            "Total sampled windows      :",
            len(self.windows)
        )

        print(
            "\nPrivacy dataset preloaded "
            "successfully."
        )

    def __len__(self):

        return len(
            self.windows
        )

    def __getitem__(
        self,
        index
    ):

        if index < 0:
            index += len(
                self.windows
            )

        if (
            index < 0
            or index >= len(
                self.windows
            )
        ):
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        # --------------------------------------------------
        # NO np.load() HERE.
        # Everything is already in RAM.
        # --------------------------------------------------

        window = (
            self.windows[index]
        )

        activity = int(
            self.activities[index]
        )

        gender = int(
            self.genders[index]
        )

        weight = int(
            self.weights[index]
        )

        window_tensor = (
            torch.from_numpy(
                window
            ).unsqueeze(0)
        )

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

    # ==========================================================
# Full Balanced Privacy Dataset and DataLoaders
# Narval GPU Experiment
# ==========================================================

print("\n" + "=" * 75)
print("PRIVACY DATASET / DATALOADER CONFIGURATION")
print("=" * 75)

if RUN_MODE == "full_gpu":

    # ------------------------------------------------------
    # Full balanced training dataset
    # ------------------------------------------------------
    # Gender dataset:
    #   20 subjects from Gender 0
    #   20 subjects from Gender 1
    #
    # 40 subjects × 6 activities × 500 windows
    # = approximately 120,000 training windows
    # ------------------------------------------------------

    full_privacy_train_dataset = BalancedPrivacyIMUDataset(
        processed_dir=PROCESSED_DIR,
        split="train",
        private_attribute=PRIVATE_ATTRIBUTE,
        samples_per_activity_per_subject=500,
        seed=42
    )

    # ------------------------------------------------------
    # Full balanced testing dataset
    # ------------------------------------------------------
    # Gender dataset:
    #   4 subjects from Gender 0
    #   4 subjects from Gender 1
    #
    # 8 subjects × 6 activities × 250 windows
    # = approximately 12,000 testing windows
    # ------------------------------------------------------

    full_privacy_test_dataset = BalancedPrivacyIMUDataset(
        processed_dir=PROCESSED_DIR,
        split="test",
        private_attribute=PRIVATE_ATTRIBUTE,
        samples_per_activity_per_subject=250,
        seed=43
    )

    # ------------------------------------------------------
    # Full training DataLoader
    # ------------------------------------------------------

    full_privacy_train_loader = DataLoader(
        full_privacy_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # ------------------------------------------------------
    # Full testing DataLoader
    # ------------------------------------------------------

    full_privacy_test_loader = DataLoader(
        full_privacy_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ------------------------------------------------------
    # Select the full loaders for the Narval experiment
    # ------------------------------------------------------

    privacy_training_loader = full_privacy_train_loader
    privacy_testing_loader = full_privacy_test_loader

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 75)
    print("FULL GPU PRIVACY DATASET READY")
    print("=" * 75)

    print(
        "Private attribute :",
        PRIVATE_ATTRIBUTE
    )

    print(
        "Training samples  :",
        len(full_privacy_train_dataset)
    )

    print(
        "Testing samples   :",
        len(full_privacy_test_dataset)
    )

    print(
        "Training batches  :",
        len(full_privacy_train_loader)
    )

    print(
        "Testing batches   :",
        len(full_privacy_test_loader)
    )

    print(
        "Batch size        :",
        BATCH_SIZE
    )

    print(
        "Workers           :",
        0
    )

    print(
        "Pinned memory     :",
        True
    )

    print("=" * 75)


else:

    # ------------------------------------------------------
    # Small CPU / local verification experiment
    # ------------------------------------------------------

    privacy_training_loader = privacy_train_loader
    privacy_testing_loader = privacy_test_loader

    print("\n" + "=" * 75)
    print("SMALL CPU PRIVACY DATASET READY")
    print("=" * 75)

    print(
        "Private attribute :",
        PRIVATE_ATTRIBUTE
    )

    print(
        "Training samples  :",
        len(privacy_train_dataset)
    )

    print(
        "Testing samples   :",
        len(privacy_test_dataset)
    )

    print(
        "Training batches  :",
        len(privacy_train_loader)
    )

    print(
        "Testing batches   :",
        len(privacy_test_loader)
    )

    print(
        "Batch size        :",
        BATCH_SIZE
    )

    print("=" * 75)


# In[11]:


# ==========================================================
# Cell 3 - Privacy Classifier for CPU and GPU
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivacyClassifier(nn.Module):
    """
    Lightweight CNN for private-attribute prediction.

    Input:
        IMU windows with shape:
        (batch_size, 1, 128, 30)

    Outputs:
        private_logits:
            Raw logits for gender or weight prediction.

        z_private:
            Private representation with dimension Z_DIM.
    """

    def __init__(
        self,
        num_private_classes,
        z_dim
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
            out_features=num_private_classes
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

        private_logits = self.private_classifier(
            z_private
        )

        return private_logits, z_private


# ==========================================================
# Create Privacy Model
# ==========================================================

privacy_model = PrivacyClassifier(
    num_private_classes=NUM_PRIVATE_CLASSES,
    z_dim=Z_DIM
).to(device)


print("=" * 70)
print("PRIVACY CLASSIFIER")
print("=" * 70)

print("Run mode          :", RUN_MODE)
print("Device            :", device)
print("Private attribute :", PRIVATE_ATTRIBUTE)
print("Private classes   :", NUM_PRIVATE_CLASSES)
print("Embedding size    :", Z_DIM)


# ==========================================================
# Select Verification Loader
# ==========================================================

if RUN_MODE == "small_cpu":

    verification_loader = privacy_train_loader

else:

    # For full GPU mode, this loader must be created from
    # the full multi-subject privacy dataset.
    verification_loader = full_privacy_train_loader


# ==========================================================
# Verify One Forward Pass
# ==========================================================

privacy_windows, privacy_activity, privacy_gender, privacy_weight = next(
    iter(verification_loader)
)

privacy_model.eval()

with torch.no_grad():

    privacy_windows_device = privacy_windows.to(
        device,
        dtype=torch.float32,
        non_blocking=True
    )

    test_private_logits, test_private_embedding = privacy_model(
        privacy_windows_device
    )


assert test_private_logits.shape == (
    privacy_windows.shape[0],
    NUM_PRIVATE_CLASSES
)

assert test_private_embedding.shape == (
    privacy_windows.shape[0],
    Z_DIM
)


print("\n" + "=" * 70)
print("PRIVACY MODEL OUTPUT VERIFICATION")
print("=" * 70)

print(
    "Input window shape      :",
    tuple(privacy_windows.shape)
)

print(
    "Private logits shape    :",
    tuple(test_private_logits.shape)
)

print(
    "Private embedding shape :",
    tuple(test_private_embedding.shape)
)

print(
    "\nPrivacy classifier verified successfully."
)


# In[12]:


# ==========================================================
# Cell 4 - Train Privacy Classifier
# CPU and GPU/Narval Compatible
# ==========================================================

import time
from pathlib import Path

import torch
import torch.nn as nn


# ----------------------------------------------------------
# Select training loader
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    privacy_training_loader = privacy_train_loader

else:

    # This loader must be created from the full
    # multi-subject privacy dataset before full GPU training.
    privacy_training_loader = full_privacy_train_loader


# ----------------------------------------------------------
# Loss function and optimizer
# ----------------------------------------------------------

privacy_criterion = nn.CrossEntropyLoss()

privacy_optimizer = torch.optim.Adam(
    privacy_model.parameters(),
    lr=LEARNING_RATE
)

PRIVACY_EPOCHS = EPOCHS


# ----------------------------------------------------------
# Store training history
# ----------------------------------------------------------

privacy_training_losses = []
privacy_training_accuracies = []

start_time = time.time()


print("=" * 75)
print("PRIVACY CLASSIFIER TRAINING")
print("=" * 75)

print("Run mode          :", RUN_MODE)
print("Device            :", device)
print("Private attribute :", PRIVATE_ATTRIBUTE)
print("Epochs            :", PRIVACY_EPOCHS)
print("Learning rate     :", LEARNING_RATE)
print("Batch size        :", BATCH_SIZE)


# ----------------------------------------------------------
# Training loop
# ----------------------------------------------------------

for epoch in range(PRIVACY_EPOCHS):

    privacy_model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for windows, activity, gender, weight in privacy_training_loader:

        windows = windows.to(
            device,
            dtype=torch.float32,
            non_blocking=True
        )

        # --------------------------------------------------
        # Select the private target
        # --------------------------------------------------

        if PRIVATE_ATTRIBUTE == "gender":
            private_labels = gender

        else:
            private_labels = weight


        # Small CPU datasets currently return one-hot labels.
        # Full GPU datasets may return integer class labels.
        if private_labels.ndim > 1:

            private_targets = torch.argmax(
                private_labels,
                dim=1
            )

        else:

            private_targets = private_labels


        private_targets = private_targets.to(
            device,
            dtype=torch.long,
            non_blocking=True
        )


        # --------------------------------------------------
        # Forward and backward pass
        # --------------------------------------------------

        privacy_optimizer.zero_grad(
            set_to_none=True
        )

        private_logits, z_private = privacy_model(
            windows
        )

        loss = privacy_criterion(
            private_logits,
            private_targets
        )

        loss.backward()
        privacy_optimizer.step()


        # --------------------------------------------------
        # Training statistics
        # --------------------------------------------------

        batch_size = private_targets.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predicted_private_classes = torch.argmax(
            private_logits,
            dim=1
        )

        correct_predictions += (
            predicted_private_classes
            == private_targets
        ).sum().item()

        total_samples += batch_size


    epoch_loss = (
        running_loss / total_samples
    )

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    privacy_training_losses.append(
        epoch_loss
    )

    privacy_training_accuracies.append(
        epoch_accuracy
    )

    print(
        f"Epoch {epoch + 1:02d}/{PRIVACY_EPOCHS:02d} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Training Accuracy: {epoch_accuracy:.4f}"
    )


# ----------------------------------------------------------
# Training summary
# ----------------------------------------------------------

training_time = (
    time.time() - start_time
)

print("\n" + "=" * 75)
print("PRIVACY CLASSIFIER TRAINING COMPLETED")
print("=" * 75)

print(
    f"Final training loss     : "
    f"{privacy_training_losses[-1]:.4f}"
)

print(
    f"Final training accuracy : "
    f"{privacy_training_accuracies[-1]:.4f}"
)

print(
    f"Total training time     : "
    f"{training_time:.2f} seconds"
)


# ----------------------------------------------------------
# Save model checkpoint
# ----------------------------------------------------------

model_filename = (
    f"{RUN_MODE}_privacy_"
    f"{PRIVATE_ATTRIBUTE}_model.pt"
)

MODEL_SAVE_PATH = Path(
    model_filename
)

checkpoint = {
    "model_state_dict":
        privacy_model.state_dict(),

    "optimizer_state_dict":
        privacy_optimizer.state_dict(),

    "training_losses":
        privacy_training_losses,

    "training_accuracies":
        privacy_training_accuracies,

    "epochs":
        PRIVACY_EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "private_attribute":
        PRIVATE_ATTRIBUTE,

    "num_private_classes":
        NUM_PRIVATE_CLASSES,

    "z_dim":
        Z_DIM,

    "run_mode":
        RUN_MODE
}

torch.save(
    checkpoint,
    MODEL_SAVE_PATH
)

print(
    "\nPrivacy model checkpoint saved as:",
    MODEL_SAVE_PATH
)


# In[13]:


# ==========================================================
# Cell 5 - Evaluate Privacy Classifier
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ----------------------------------------------------------
# Select evaluation loader and dataset sizes
# ----------------------------------------------------------

if RUN_MODE == "small_cpu":

    privacy_evaluation_loader = privacy_test_loader

    training_sample_count = len(
        privacy_train_dataset
    )

    testing_sample_count = len(
        privacy_test_dataset
    )

else:

    privacy_evaluation_loader = (
        full_privacy_test_loader
    )

    training_sample_count = len(
        full_privacy_train_loader.dataset
    )

    testing_sample_count = len(
        full_privacy_test_loader.dataset
    )


# ----------------------------------------------------------
# Select class names
# ----------------------------------------------------------

if PRIVATE_ATTRIBUTE == "gender":

    private_class_names = [
        "Gender 0",
        "Gender 1"
    ]

    private_class_ids = [
        0,
        1
    ]

else:

    private_class_names = [
        "Weight Class 0",
        "Weight Class 1",
        "Weight Class 2"
    ]

    private_class_ids = [
        0,
        1,
        2
    ]


# ----------------------------------------------------------
# Run evaluation
# ----------------------------------------------------------

privacy_model.eval()

true_private_labels = []
predicted_private_labels = []

with torch.no_grad():

    for windows, activity, gender, weight in (
        privacy_evaluation_loader
    ):

        windows = windows.to(
            device,
            dtype=torch.float32,
            non_blocking=True
        )

        # Select the sensitive target
        if PRIVATE_ATTRIBUTE == "gender":
            private_labels = gender
        else:
            private_labels = weight

        # Small CPU datasets use one-hot labels.
        # Full GPU datasets may use integer labels.
        if private_labels.ndim > 1:

            private_targets = torch.argmax(
                private_labels,
                dim=1
            )

        else:

            private_targets = private_labels

        private_targets = private_targets.to(
            device,
            dtype=torch.long,
            non_blocking=True
        )

        private_logits, z_private = privacy_model(
            windows
        )

        predicted_classes = torch.argmax(
            private_logits,
            dim=1
        )

        true_private_labels.extend(
            private_targets.cpu().numpy()
        )

        predicted_private_labels.extend(
            predicted_classes.cpu().numpy()
        )


true_private_labels = np.asarray(
    true_private_labels
)

predicted_private_labels = np.asarray(
    predicted_private_labels
)


# ----------------------------------------------------------
# Calculate evaluation metrics
# ----------------------------------------------------------

privacy_accuracy = accuracy_score(
    true_private_labels,
    predicted_private_labels
)

privacy_precision = precision_score(
    true_private_labels,
    predicted_private_labels,
    average="macro",
    zero_division=0
)

privacy_recall = recall_score(
    true_private_labels,
    predicted_private_labels,
    average="macro",
    zero_division=0
)

privacy_macro_f1 = f1_score(
    true_private_labels,
    predicted_private_labels,
    average="macro",
    zero_division=0
)

privacy_weighted_f1 = f1_score(
    true_private_labels,
    predicted_private_labels,
    average="weighted",
    zero_division=0
)


# ----------------------------------------------------------
# Print overall results
# ----------------------------------------------------------

print("=" * 75)
print("PRIVACY CLASSIFICATION RESULTS")
print("=" * 75)

print("Run mode            :", RUN_MODE)
print("Device              :", device)
print("Private attribute   :", PRIVATE_ATTRIBUTE)
print("Training samples    :", training_sample_count)
print("Testing samples     :", testing_sample_count)
print(f"Test Accuracy       : {privacy_accuracy:.4f}")
print(f"Macro Precision     : {privacy_precision:.4f}")
print(f"Macro Recall        : {privacy_recall:.4f}")
print(f"Macro F1-score      : {privacy_macro_f1:.4f}")
print(f"Weighted F1-score   : {privacy_weighted_f1:.4f}")


# ----------------------------------------------------------
# Classification report
# ----------------------------------------------------------

print("\n" + "=" * 75)
print("PER-CLASS PRIVACY CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        true_private_labels,
        predicted_private_labels,
        labels=private_class_ids,
        target_names=private_class_names,
        digits=4,
        zero_division=0
    )
)


# ----------------------------------------------------------
# Plain confusion matrix
# ----------------------------------------------------------

privacy_cm = confusion_matrix(
    true_private_labels,
    predicted_private_labels,
    labels=private_class_ids
)

confusion_matrix_table = pd.DataFrame(
    privacy_cm,
    index=[
        f"True {name}"
        for name in private_class_names
    ],
    columns=[
        f"Predicted {name}"
        for name in private_class_names
    ]
)

print("=" * 85)
print("PRIVACY CONFUSION MATRIX")
print("=" * 85)

print(
    confusion_matrix_table.to_string()
)


# ----------------------------------------------------------
# Chance-level reference
# ----------------------------------------------------------

chance_accuracy = (
    1.0 / NUM_PRIVATE_CLASSES
)

print(
    f"\nChance-level accuracy : "
    f"{chance_accuracy:.4f}"
)


# In[ ]:





# In[ ]:




