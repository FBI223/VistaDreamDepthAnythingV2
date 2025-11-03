import os
import numpy as np
import torch
import cv2
from depth_anything_v2.dpt import DepthAnythingV2


class Depth_Pro_Tool:
    """
    Wrapper dla Depth Anything V2 — kompatybilny z VistaDream.
    Zwraca znormalizowaną mapę głębokości (0–1), odwracając inverse depth.
    """

    def __init__(
        self,
        device: str | None = None,
        encoder: str = "vitl",
        ckpt_path: str = "/checkpoints/depth_anything_v2_vitl.pth",
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"[DepthAnythingV2] Using device: {self.device}")

        self.model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        if encoder not in self.model_configs:
            raise ValueError(f"Nieznany encoder '{encoder}'. Dozwolone: {list(self.model_configs.keys())}")

        self.model = DepthAnythingV2(**self.model_configs[encoder])

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"❌ Checkpoint nie znaleziony: {ckpt_path}\n"
                f"Upewnij się, że pobrałeś model Depth Anything V2-Large do katalogu checkpoints/"
            )

        print(f"[DepthAnythingV2] Loading weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.ndim == 3 and image.shape[-1] == 3 and image[..., 0].mean() > image[..., 2].mean():
            image = image[..., ::-1]
        if image.max() > 1.1:
            image = image / 255.0
        return (image * 255.0).astype(np.uint8)

    def __call__(self, image: np.ndarray, f_px=None) -> np.ndarray:
        image = self.preprocess(image)
        inv_depth = self.model.infer_image(image)
        inv_depth = np.clip(inv_depth, 1e-6, None)
        depth = 1.0 / inv_depth
        depth = depth / (depth.max() + 1e-8)
        depth = np.clip(depth, 0.0, 1.0)
        return depth.astype(np.float32)