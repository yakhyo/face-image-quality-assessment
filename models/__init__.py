from .backbones import IResNet18, IResNet50, IResNet100, MobileFaceNet
from .ediffiqa import MLP, eDifFIQA
from .ediffiqa_onnx import eDifFIQAOnnx

__all__ = ["IResNet100", "IResNet18", "IResNet50", "MLP", "MobileFaceNet", "eDifFIQA", "eDifFIQAOnnx", "get_model"]


BACKBONES = {
    "t": MobileFaceNet,
    "s": IResNet18,
    "m": IResNet50,
    "l": IResNet100,
}


def get_model(variant: str, return_feat: bool = False) -> eDifFIQA:
    """Construct an untrained eDifFIQA model for the given variant ("t"/"s"/"m"/"l")."""
    key = variant.lower()
    if key not in BACKBONES:
        raise ValueError(f"Unknown variant '{variant}'. Valid: {', '.join(BACKBONES)}.")
    return eDifFIQA(backbone=BACKBONES[key](), head=MLP(), return_feat=return_feat)
