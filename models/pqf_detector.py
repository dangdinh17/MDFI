"""BiLSTM detector trained from feature-distance PQF labels."""

import torch
import torch.nn as nn


class FeaturePQFDetector(nn.Module):
    def __init__(self, input_size=1152, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features):
        encoded, _ = self.encoder(features)
        return self.classifier(encoded).squeeze(-1)

    @staticmethod
    def local_peak_labels(probabilities, threshold=0.5):
        """Convert probabilities to non-adjacent temporal peak labels."""
        labels = torch.zeros_like(probabilities, dtype=torch.bool)
        if probabilities.shape[-1] >= 3:
            center = probabilities[..., 1:-1]
            labels[..., 1:-1] = (
                (center >= threshold)
                & (center > probabilities[..., :-2])
                & (center > probabilities[..., 2:])
            )
        return labels
