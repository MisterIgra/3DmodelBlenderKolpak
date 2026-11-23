from __future__ import annotations

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from .constants import CENTER_TYPES, FAMILIES, FINISHES, SPOKE_COUNTS, REGRESSION_FIELDS


class HubcapNet(nn.Module):
    """Pretrained ResNet-18 with independent heads for visible attributes and ratios."""
    architecture = "resnet18_multitask_v2"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            backbone = resnet18(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Не удалось загрузить pretrained-веса ResNet18. Проверьте интернет/кэш "
                "torchvision или выставьте pretrained: false."
            ) from exc
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.20)

        def head(out_features: int):
            return nn.Sequential(
                nn.Linear(features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(256, out_features),
            )

        self.family_head = head(len(FAMILIES))
        self.spokes_head = head(len(SPOKE_COUNTS))
        self.center_head = head(len(CENTER_TYPES))
        self.finish_head = head(len(FINISHES))
        self.regression_head = nn.Sequential(head(len(REGRESSION_FIELDS)), nn.Tanh())

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return {
            "family": self.family_head(features),
            "spokes": self.spokes_head(features),
            "center": self.center_head(features),
            "finish": self.finish_head(features),
            "regression": self.regression_head(features),
        }
