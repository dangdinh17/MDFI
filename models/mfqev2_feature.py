"""PyTorch MFQE 2.0 enhancement network with feature-derived PQF routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(image, flow):
    """Warp ``image`` using pixel-space flow shaped Bx2xHxW."""
    batch, _, height, width = image.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    norm_x = flow[:, 0] * (2.0 / max(width - 1, 1))
    norm_y = flow[:, 1] * (2.0 / max(height - 1, 1))
    grid = grid + torch.stack((norm_x, norm_y), dim=-1)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


class LearnedFrameAligner(nn.Module):
    """Compact end-to-end motion compensation used before MFQE fusion."""

    def __init__(self, max_displacement=12.0):
        super().__init__()
        self.max_displacement = float(max_displacement)
        self.flow = nn.Sequential(
            nn.Conv2d(2, 32, 7, padding=3),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.PReLU(32),
            nn.Conv2d(32, 24, 3, padding=1),
            nn.PReLU(24),
            nn.Conv2d(24, 16, 3, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, 2, 3, padding=1),
        )
        nn.init.zeros_(self.flow[-1].weight)
        nn.init.zeros_(self.flow[-1].bias)

    def forward(self, reference, target):
        flow = torch.tanh(self.flow(torch.cat((reference, target), dim=1)))
        return flow_warp(reference, flow * self.max_displacement)


class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )


class MFQEEnhancer(nn.Module):
    """One MF-CNN branch for either PQF or non-PQF targets."""

    def __init__(self, num_features=32, max_displacement=12.0):
        super().__init__()
        self.aligner = LearnedFrameAligner(max_displacement=max_displacement)
        self.ks3 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(1, num_features, 3, padding=1), nn.PReLU(num_features)) for _ in range(3)]
        )
        self.ks5 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(1, num_features, 5, padding=2), nn.PReLU(num_features)) for _ in range(3)]
        )
        self.ks7 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(1, num_features, 7, padding=3), nn.PReLU(num_features)) for _ in range(3)]
        )
        self.rec1 = ConvBNPReLU(9 * num_features, num_features)
        self.rec2 = ConvBNPReLU(num_features, num_features)
        self.rec3 = ConvBNPReLU(2 * num_features, num_features)
        self.rec4 = ConvBNPReLU(3 * num_features, num_features)
        self.rec5 = ConvBNPReLU(4 * num_features, num_features)
        self.out = nn.Conv2d(num_features, 1, 3, padding=1)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[1:3] != (3, 1):
            raise ValueError(f"Expected Bx3x1xHxW input, received {frames.shape}")
        center = frames[:, 1]
        aligned = [
            self.aligner(frames[:, 0], center),
            center,
            self.aligner(frames[:, 2], center),
        ]
        features = []
        for index, frame in enumerate(aligned):
            features.extend((self.ks3[index](frame), self.ks5[index](frame), self.ks7[index](frame)))
        merged = torch.cat(features, dim=1)
        dense = [self.rec1(merged)]
        dense.append(self.rec2(dense[-1]))
        dense.append(self.rec3(torch.cat(dense, dim=1)))
        dense.append(self.rec4(torch.cat(dense, dim=1)))
        reconstructed = self.rec5(torch.cat(dense, dim=1))
        return center + self.out(reconstructed)


class FeaturePQFMFQEv2(nn.Module):
    """Separate MFQE branches for PQF and non-PQF enhancement."""

    def __init__(self, num_features=32, max_displacement=12.0):
        super().__init__()
        self.non_pqf = MFQEEnhancer(num_features, max_displacement)
        self.pqf = MFQEEnhancer(num_features, max_displacement)

    def forward(self, frames, is_pqf):
        is_pqf = is_pqf.reshape(-1).bool()
        output = torch.zeros_like(frames[:, 1])
        non_pqf_indices = (~is_pqf).nonzero(as_tuple=False).flatten()
        pqf_indices = is_pqf.nonzero(as_tuple=False).flatten()
        if non_pqf_indices.numel():
            prediction = self.non_pqf(frames.index_select(0, non_pqf_indices))
            output = output.index_copy(0, non_pqf_indices, prediction)
        if pqf_indices.numel():
            prediction = self.pqf(frames.index_select(0, pqf_indices))
            output = output.index_copy(0, pqf_indices, prediction)
        return output
