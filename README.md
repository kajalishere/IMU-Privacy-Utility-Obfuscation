# IMU Privacy–Utility Obfuscation

An experimental framework for wearable IMU data obfuscation that investigates
the privacy–utility trade-off between preserving human activity recognition
and reducing sensitive-attribute inference.

The project studies whether wearable inertial sensor signals can be transformed
so that useful activity-related information is retained while information
associated with a sensitive demographic attribute is made more difficult to
infer.

In the current experiments:

- **Utility / public attribute:** Human activity
- **Sensitive / private attribute:** Gender
- **Input modality:** Wearable IMU signals
- **Obfuscation approach:** Conditional diffusion with privacy-guided and
  adversarial extensions


## Project Overview

Wearable inertial sensors provide useful information for applications such as
Human Activity Recognition (HAR). However, the same sensor measurements may
also contain information that can be exploited to infer attributes that are not
required for the intended application.

This project experimentally investigates this privacy–utility problem.

The overall objective is:

> Preserve activity-recognition utility while reducing the ability of an
> inference model to recover gender information from transformed IMU signals.

The work started from a conditional diffusion-based sensor obfuscation
framework inspired by PrivDiffuser and was subsequently extended through
privacy-guided fine-tuning, restricted-timestep training, and an adaptive
adversarial privacy component.


## Dataset

The experiments use the **Daily Activities Wearable Dataset for
Cardiorespiratory Fitness Estimation**, published on Zenodo in 2025.

**Dataset DOI:** `10.5281/zenodo.15857137`

The dataset contains synchronized wearable measurements collected during
structured daily activities. Available information includes:

- Accelerometer measurements
- Gyroscope measurements
- Quaternion orientation measurements
- Activity annotations
- Participant demographic information
- Heart-rate and SpO2 measurements

This project focuses on the **IMU measurements, activity labels, and gender
attribute**. Physiological measurements are not used in the current
privacy–utility experiments.

### Experimental representation

The processed IMU windows used by the models have:

- **Window length:** 128 samples
- **Number of sensor features:** 30
- **Model input shape:** `(1, 128, 30)`
- **Activity classes:** 6
- **Gender classes:** 2

The raw dataset is **not redistributed in this repository**. It should be
downloaded directly from the original Zenodo source and processed using the
notebooks provided in this repository.

### Hybrid Adversarial Extension

The final experimental configuration extends privacy-guided diffusion with an
adaptive gender adversary.

The adversary learns to infer gender from transformed IMU signals while the
diffusion model is optimized in the opposing direction. This creates an
adversarial learning signal intended to make the transformation more robust
against sensitive-attribute inference.

The approach combines:

- Activity-conditioned diffusion
- Explicit activity-utility guidance
- Sensitive-attribute privacy guidance
- Restricted diffusion timesteps
- Adaptive adversarial gender inference

The adversarial component is conceptually motivated by adversarial sensor-data
sanitization approaches such as DySan, while the diffusion component is
motivated by PrivDiffuser.


## Experimental Configurations

Several configurations were investigated during development, including:

- Baseline conditional diffusion
- Privacy-guided diffusion
- Different privacy and utility loss weights
- Restricted-timestep privacy-guided training
- Hybrid adversarial privacy-guided diffusion

For the final hybrid experiment:

- **Training timestep range:** `5–40`
- **Utility weight (λ_u):** `2`
- **Privacy weight (λ_p):** `100`
- **Evaluation timesteps:** `5, 10, 20, 30, 40`


## Selected Privacy–Utility Operating Point

Among the evaluated timesteps, **t = 10** provides the selected
privacy–utility operating point for the final hybrid configuration.

Compared with the balanced raw-data baseline:

| Metric | Raw | Hybrid (t=10) | Change |
|--------|----:|--------------:|-------:|
| Activity Accuracy | 74.17% | **70.42%** | **−3.75 percentage points** |
| Gender Accuracy | 70.83% | **48.75%** | **−22.08 percentage points** |

Thus, the selected configuration substantially reduces gender inference while
limiting the corresponding decrease in activity-recognition accuracy.

This result should be interpreted as an **empirical privacy–utility trade-off
for the evaluated dataset, classifiers, and experimental configuration**, not
as a general guarantee of privacy.

