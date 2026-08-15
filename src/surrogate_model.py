"""
==========================================================
Surrogate Utility Classifier
PrivDiffuser
==========================================================

This module implements the surrogate activity classifier
used by PrivDiffuser.

Input
-----
(1, 128, 30) IMU window

Output
------
Activity logits
60-dimensional latent embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Surrogate_Utility_Classifier(nn.Module):

    def __init__(self, num_classes=6, z_dim=60):

        super(Surrogate_Utility_Classifier, self).__init__()

        self.flatten = nn.Flatten()

        # ----------------------------------------
        # Feature Extractor
        # ----------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=2,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            stride=1,
            padding=1
        )

        # Feature Projection

        self.fc1 = nn.Linear(66560, 512)

        self.fc2 = nn.Linear(512, 128)

        self.fc3 = nn.Linear(128, z_dim)

        self.fc4 = nn.Linear(z_dim, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        embedding = self.relu(self.fc3(x))

        embedding = F.normalize(
            embedding,
            dim=-1
        )

        logits = self.fc4(embedding)

        return logits, embedding