
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

import os
import torch

# 🔧 Wymuś działanie tylko na CPU (blokuje CUDA całkowicie)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda : False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
# ────────────────────────────────────────────────────────────────
# ⚙️ Adapter: Depth Anything → pseudo-metry (kompatybilny z Depth Pro)
# ────────────────────────────────────────────────────────────────
class DepthAnythingAdapterMetric:
    """
    Adapter kompatybilny z Apple Depth Pro:
        depth, intrinsic = adapter.infer(image, f_px=None)
        depth  → mapa głębokości (float32, pseudo-metry 0.5–20 m)
        intrinsic → macierz kamery 3×3 (np.eye(3) lub z f_px)
    """

    def __init__(self, checkpoint_path="checkpoints/depth_anything_v2_vitl.pth",
                 device="cpu", depth_range=(0.5, 20.0)):
        self.device = torch.device(device)
        self.depth_min, self.depth_max = depth_range

        cfg = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }

        print(f"[INFO] Loading DepthAnythingV2 ({device}), "
              f"pseudo-metry range {self.depth_min}–{self.depth_max} m")

        self.model = DepthAnythingV2(**cfg)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()


    def infer(self, image: np.ndarray, f_px=None):
        """Zwraca: depth_map [H×W] i intrinsic [3×3]."""
        if image.shape[-1] == 3 and np.mean(image[..., 0]) > np.mean(image[..., 2]):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth_rel = self.model.infer_image(image)
            if isinstance(depth_rel, torch.Tensor):
                depth_rel = depth_rel.detach().cpu().numpy()

        # Normalizacja 0–1
        depth_rel = (depth_rel - depth_rel.min()) / (depth_rel.max() - depth_rel.min() + 1e-8)
        # Skalowanie do pseudo-metrycznego zakresu
        depth_m = self.depth_min + (self.depth_max - self.depth_min) * depth_rel

        # ⬅️ ODWRÓCENIE MAPY GŁĘBOKOŚCI
        depth_m = depth_m.max() - depth_m

        # Macierz intrinsics (zgodna z Depth Pro)
        intrinsic = np.eye(3, dtype=np.float32)
        if f_px is not None:
            intrinsic[0, 0] = intrinsic[1, 1] = f_px
        else:
            intrinsic[0, 0] = intrinsic[1, 1] = 1000.0  # domyślna ogniskowa

        return depth_m.astype(np.float32), intrinsic




if __name__ == "__main__":
    adapter = DepthAnythingAdapterMetric(
        "checkpoints/depth_anything_v2_vitl.pth",
        device="cpu",          # lub "cuda"
        depth_range=(0.5, 20.0)
    )

    img = cv2.imread("WAWEL_MODIFIED/2/2.jpg")
    depth, intrinsic = adapter.infer(img)

    print(f"[RESULT] Zakres pseudo-metryczny: {depth.min():.3f} m → {depth.max():.3f} m")
    print(f"[RESULT] Intrinsic:\n{intrinsic}")

    cv2.imwrite("depth_anything_metric.png", (depth / depth.max() * 255).astype(np.uint8))

