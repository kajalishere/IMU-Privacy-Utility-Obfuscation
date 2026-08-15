"""
==========================================================
Auxiliary Privacy Classifier
PrivDiffuser
==========================================================

This module implements the auxiliary privacy classifier
used by PrivDiffuser.

Input
-----
(1,128,30) IMU window
60-dimensional embedding from surrogate model

Output
------
Gender prediction logits
Privacy latent representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Aux_Priv_Classifier(nn.Module):

    def __init__(self, num_classes=2, z_dim=60):

        super(Aux_Priv_Classifier, self).__init__()

        self.flatten = nn.Flatten()

        # --------------------------------------------------
        # Feature Extractor
        # --------------------------------------------------

        self.fc1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=2,
            stride=1,
            padding=1
        )

        self.fc2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            stride=1,
            padding=1
        )

        # Concatenate flattened features with surrogate embedding

        self.fc3 = nn.Linear(
            66560 + z_dim,
            512
        )

        self.fc4 = nn.Linear(512, 128)

        self.fc5 = nn.Linear(128, z_dim)

        self.fc6 = nn.Linear(z_dim, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, emb):

        h1 = self.relu(self.fc1(x))

        h2 = self.flatten(
            self.relu(self.fc2(h1))
        )

        h2 = torch.cat(
            (h2, emb),
            dim=1
        )

        h3 = self.relu(self.fc3(h2))

        h4 = self.relu(self.fc4(h3))

        h5 = self.relu(self.fc5(h4))

        z = F.normalize(
            h5,
            dim=-1
        )

        logits = self.fc6(z)

        return logits, z