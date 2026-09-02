"""Frozen perceptual feature metric shared by PQF labels and training loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
except ImportError as exc:  # pragma: no cover - explicit runtime guidance
    raise ImportError("Install the 'lpips' package to use feature-PQF training") from exc


class LPIPSAlexFeatureLoss(nn.Module):
    """LPIPS-Alex distance for grayscale Y images in the [0, 1] range."""

    def __init__(self, resize=None):
        super().__init__()
        self.resize = resize
        self.metric = lpips.LPIPS(net="alex", version="0.1", verbose=False)
        self.metric.eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)

    def _prepare(self, image):
        if image.ndim != 4:
            raise ValueError(f"Expected BCHW input, received shape {image.shape}")
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            raise ValueError("LPIPS inputs must have one or three channels")
        if self.resize:
            image = F.interpolate(
                image,
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            )
        return image.mul(2.0).sub(1.0)

    def forward(self, prediction, target):
        prediction = self._prepare(prediction.float())
        target = self._prepare(target.float())
        return self.metric(prediction, target, normalize=False).mean()

    @torch.no_grad()
    def distance(self, distorted, reference):
        distorted = self._prepare(distorted.float())
        reference = self._prepare(reference.float())
        return self.metric(distorted, reference, normalize=False).flatten()

    @torch.no_grad()
    def embedding(self, image):
        """Return a compact 1152-D descriptor from LPIPS AlexNet features."""
        image = self._prepare(image.float())
        if self.metric.version == "0.1":
            image = self.metric.scaling_layer(image)
        feature_maps = self.metric.net.forward(image)
        pooled = []
        for feature_map in feature_maps:
            normalized = lpips.normalize_tensor(feature_map)
            pooled.append(normalized.mean(dim=(-2, -1)))
        return torch.cat(pooled, dim=1)

    @torch.no_grad()
    def distance_and_embedding(self, distorted, reference):
        """Compute distance and distorted-frame descriptor in two forwards."""
        distorted = self._prepare(distorted.float())
        reference = self._prepare(reference.float())
        if self.metric.version == "0.1":
            distorted = self.metric.scaling_layer(distorted)
            reference = self.metric.scaling_layer(reference)
        distorted_maps = self.metric.net.forward(distorted)
        reference_maps = self.metric.net.forward(reference)
        distances = 0.0
        pooled = []
        for index, (distorted_map, reference_map) in enumerate(
            zip(distorted_maps, reference_maps)
        ):
            distorted_feature = lpips.normalize_tensor(distorted_map)
            reference_feature = lpips.normalize_tensor(reference_map)
            difference = (distorted_feature - reference_feature).square()
            distances = distances + self.metric.lins[index](difference).mean(
                dim=(-2, -1), keepdim=True
            )
            pooled.append(distorted_feature.mean(dim=(-2, -1)))
        return distances.flatten(), torch.cat(pooled, dim=1)
