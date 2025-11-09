import os
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# ───────── konfiguracja modelu ─────────
cfgs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
encoder = "vitl"
ckpt = "checkpoints/depth_anything_v2_vitl.pth"

print(f"[INIT] Wczytywanie modelu Depth Anything V2 (relatywnego): {ckpt}")
model = DepthAnythingV2(**cfgs[encoder])
state = torch.load(ckpt, map_location="cpu")
model.load_state_dict(state)
model = model.to("cpu").eval()
print("[OK] Model gotowy.\n")

# ───────── pętla po folderach ─────────
for i in range(5, 7):
    folder = os.path.join("WAWEL_MODIFIED", str(i))
    img_path = os.path.join(folder, f"{i}.jpg")

    if not os.path.exists(img_path):
        print(f"[WARN] Pomijam brakujący plik: {img_path}")
        continue

    print(f"\n[INFO] --- Przetwarzanie: {img_path} ---")

    # --- 1️⃣ Wczytaj obraz ---
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Nie można wczytać {img_path}")
        continue

    print(f"[LOAD] Obraz załadowany:")
    print(f"       typ: {type(img)}, shape: {img.shape}, dtype: {img.dtype}")
    print(f"       kanały: B={np.mean(img[...,0]):.1f}, G={np.mean(img[...,1]):.1f}, R={np.mean(img[...,2]):.1f}")

    # --- 2️⃣ Przygotowanie wejścia ---
    # Model sam wykonuje resize / normalizację wewnętrznie
    # ale można dodać kontrolę RGB
    if np.mean(img[..., 0]) > np.mean(img[..., 2]):
        print("       Wykryto BGR (OpenCV) → konwersja na RGB")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("       Obraz już w RGB")
        img_rgb = img

    # --- 3️⃣ Predykcja ---
    with torch.no_grad():
        depth = model.infer_image(img_rgb)

    print(f"[MODEL] Wynik inferencji:")
    print(f"       typ: {type(depth)}, shape: {np.array(depth).shape}, dtype: {np.array(depth).dtype}")
    print(f"       zakres: min={depth.min():.4f}, max={depth.max():.4f}")
    if isinstance(depth, torch.Tensor):
        print(f"       urządzenie: {depth.device}")

    # --- 4️⃣ Normalizacja i zapis ---
    dmin, dmax = depth.min(), depth.max()
    depth_norm = (depth - dmin) / (dmax - dmin + 1e-8)
    depth_img = (depth_norm * 255).astype(np.uint8)
    out_path = os.path.join(folder, "depth_rel.png")
    cv2.imwrite(out_path, depth_img)

    print(f"[SAVE] Zapisano mapę głębokości → {out_path}")
    print(f"       format: uint8, zakres 0–255")
    print(f"       Zakres oryginalny (relatywny): {dmin:.4f} → {dmax:.4f}")

print("\n[DONE] Wszystkie foldery przetworzone.")
