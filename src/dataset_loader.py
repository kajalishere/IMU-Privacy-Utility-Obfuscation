"""
==============================================================
dataset_loader.py
==============================================================

Purpose
-------
This module loads the Wearable IMU Dataset into a consistent
format for the PrivDiffuser project.

Responsibilities
----------------
1. Locate the dataset directory.
2. Load participant metadata.
3. Discover available subjects.
4. Load IMU data for one participant.
5. Load IMU data for all participants.
6. Identify sensor and timestamp columns.

This module DOES NOT perform
----------------------------
- Data preprocessing
- Feature scaling
- Sliding window segmentation
- Label encoding
- Train/Test splitting

Those steps are handled separately in preprocessing.py.

Author : Kajal
Project : PrivDiffuser Dataset Adaptation
"""


from pathlib import Path
import pandas as pd

DATASET_PATH = Path("../datasets/DatasetIMUandBIOMARKERS")
METADATA_FILE = DATASET_PATH / "SubjectsInfo.xlsx"

# Function 1 : Get Dataset Path

def get_dataset_path():
    """
    Return the dataset directory.

    Returns
    -------
    pathlib.Path
        Dataset directory.

    Raises
    ------
    FileNotFoundError
        If the dataset folder does not exist.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_PATH}"
        )

    return DATASET_PATH

# Function 2 : Load Metadata
def load_metadata():
    """
    Load participant metadata.

    Returns
    -------
    pandas.DataFrame
        Metadata for all participants.
    """

    metadata_file = get_dataset_path() / "SubjectsInfo.xlsx"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_file}"
        )

    return pd.read_excel(metadata_file)


# Function 3 : Get Subject List
def get_subject_list():
    """
    Return a sorted list of all subject folders.

    Returns
    -------
    list
        Subject folder names.
    """

    dataset_path = get_dataset_path()

    subjects = sorted(
        folder.name
        for folder in dataset_path.glob("Subject*")
        if folder.is_dir()
    )

    return subjects

# Function 4 : Load IMU Data
def load_subject_imu(subject_id, sensors_only=False):
    """
    Load IMU data for one participant.

    Parameters
    ----------
    subject_id : str
        Example:
            "Subject01"

    sensors_only : bool, optional (default=False)
        False -> Return complete IMU dataframe.
        True  -> Return only sensor columns.

    Returns
    -------
    pandas.DataFrame
        IMU dataframe.
    """

    dataset_path = get_dataset_path()

    subject_number = subject_id.replace("Subject", "")

    imu_file = (
        dataset_path
        / subject_id
        / f"IMUSubject{subject_number}.csv"
    )

    if not imu_file.exists():
        raise FileNotFoundError(
            f"IMU file not found:\n{imu_file}"
        )

    imu_df = pd.read_csv(imu_file)

    if sensors_only:
        sensor_columns = get_sensor_columns(imu_df)
        imu_df = imu_df[sensor_columns]

    return imu_df

# Function 5 : Load All Subjects

def load_all_subjects(sensors_only=False):
    """
    Load IMU data for all participants.

    Parameters
    ----------
    sensors_only : bool, optional (default=False)

    Returns
    -------
    dict
        Dictionary containing IMU dataframes.
    """

    data = {}

    for subject in get_subject_list():
        data[subject] = load_subject_imu(
            subject,
            sensors_only=sensors_only
        )

    return data

# Function 6 : Get Sensor Columns
def get_sensor_columns(df):
    """
    Return sensor columns only.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list
        Accelerometer, gyroscope and quaternion columns.
    """

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith(("a_", "g_", "q_"))
    ]

    return sensor_columns

# Function 7 : Get Timestamp Columns
def get_timestamp_columns(df):
    """
    Return timestamp-related columns.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list
        Timestamp column names.
    """

    timestamp_columns = []

    for column in df.columns:

        name = column.lower()

        if (
            "time" in name
            or "timestamp" in name
            or "epoch" in name
        ):
            timestamp_columns.append(column)

    return timestamp_columns

# Test Block
if __name__ == "__main__":

    print("=" * 60)
    print("Dataset Loader Test")
    print("=" * 60)

    # Dataset Path
    print("\nDataset Path")
    print(get_dataset_path())

    # Metadata
    metadata = load_metadata()
    print("\nMetadata Shape :", metadata.shape)

    # Subjects
    subjects = get_subject_list()
    print("\nNumber of Subjects :", len(subjects))
    print("First 5 Subjects :", subjects[:5])

    # Load complete IMU data
    imu = load_subject_imu(subjects[0])

    print("\nComplete IMU Shape :", imu.shape)

    # Load sensor-only data
    sensor_imu = load_subject_imu(
        subjects[0],
        sensors_only=True
    )

    print("Sensor-only Shape :", sensor_imu.shape)

    print("\nNumber of Sensor Columns :",
          len(get_sensor_columns(imu)))

    print("Timestamp Columns :",
          get_timestamp_columns(imu))

    print("\nDataset Loader completed successfully.")