import torch
import numpy as np
import cv2
from PIL import Image
#from depth_anything_v2.dpt import DepthAnythingV2
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingProDepth:
    """
    Kompatybilny interfejs z Apple Depth Pro:
        depth, intrinsic = adapter(image, f_px=None)
    """

    def __init__(
        self,
        ckpt="checkpoints/depth_anything_v2_vitl.pth",
        encoder="vitl",
        depth_range=(0.5, 20.0),
        default_focal_px=1000.0,
        device=None
    ):
        # 🔧 automatyczny wybór CPU/GPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt = ckpt
        self.encoder = encoder
        self.depth_min, self.depth_max = depth_range
        self.default_focal_px = default_focal_px

        self._load_model()

    def _load_model(self):
        cfg = {
            "encoder": self.encoder,
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }

        print(f"[INFO] Loading DepthAnythingV2 ({self.encoder}) on {self.device} "
              f"[range {self.depth_min}–{self.depth_max} m]")

        self.model = DepthAnythingV2(**cfg)
        state = torch.load(self.ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def get_intrins(f_px, H, W):
        """Zwraca macierz intrinsics (zgodnie z Apple DepthPro)."""
        cx, cy = (W / 2.0) - 0.5, (H / 2.0) - 0.5
        return np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float32)

    def to(self, device):
        """Przenosi model między CPU ↔ GPU."""
        self.device = torch.device(device)
        self.model.to(self.device)

    def __call__(self, image, f_px=None):
        """Główne wywołanie: zwraca (depth_map, intrinsic)."""
        if isinstance(image, np.ndarray):
            if np.amax(image) <= 1.1:  # normalizowany [0–1]
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = np.array(image)

        if image.shape[-1] == 3 and np.mean(image[..., 0]) > np.mean(image[..., 2]):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # inferencja (z przeniesieniem na właściwe urządzenie)
        with torch.no_grad():
            depth_rel = self.model.infer_image(image)
            if isinstance(depth_rel, torch.Tensor):
                depth_rel = depth_rel.detach().cpu().numpy()

        # skalowanie do zakresu metrycznego
        depth_rel = (depth_rel - depth_rel.min()) / (depth_rel.max() - depth_rel.min() + 1e-8)
        depth_m = self.depth_min + (self.depth_max - self.depth_min) * depth_rel

        H, W = depth_m.shape[:2]
        f_final = f_px if f_px is not None else self.default_focal_px
        intrinsic = self.get_intrins(f_final, H, W)

        return depth_m.astype(np.float32), intrinsic



'''
# ──────────────────────────────────────────────
# 💡 Przykład użycia
# ──────────────────────────────────────────────
if __name__ == "__main__":
    adapter = DepthAnythingProDepth(
        ckpt="checkpoints/depth_anything_v2_vitl.pth",
        encoder="vitl",              # lub "vitb", "vits"
        depth_range=(0.3, 30.0),     # np. inny zakres
        default_focal_px=950.0,      # ogniskowa (px)
        device="cuda"                # lub "cpu"
    )

    img = cv2.imread("WAWEL_MODIFIED/2/2.jpg")
    depth, intr = adapter(img)

    print(f"[RESULT] Zakres: {depth.min():.3f}–{depth.max():.3f} m")
    print(f"[RESULT] Intrinsics:\n{intr}")
    cv2.imwrite("depth_anything_out.png", (depth / depth.max() * 255).astype(np.uint8))

'''
