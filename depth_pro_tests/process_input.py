import depth_pro
import cv2
import numpy as np
import torch
import os

# ───────── Inicjalizacja modelu ─────────
print("[INIT] Tworzenie modelu Apple Depth Pro...")
model, transform = depth_pro.create_model_and_transforms()
model.eval()
print("[OK] Model gotowy.\n")

base_dir = "WAWEL_MODIFIED"

for i in range(5, 7):
    folder = os.path.join(base_dir, str(i))
    img_path = os.path.join(folder, f"{i}.jpg")

    if not os.path.exists(img_path):
        print(f"[WARN] Brak pliku: {img_path}")
        continue

    print(f"\n[INFO] --- Przetwarzanie: {img_path} ---")

    # --- 1️⃣ Załaduj obraz ---
    image, _, f_px = depth_pro.load_rgb(img_path)
    print(f"[LOAD] Obraz załadowany:")
    print(f"       typ: {type(image)}, shape: {np.array(image).shape}, dtype: {np.array(image).dtype}")
    print(f"       f_px (ogniskowa w pikselach): {f_px}")

    # --- 2️⃣ Transformacja wejścia ---
    transformed = transform(image)
    print(f"[TRANSFORM] Po transformacji:")
    if isinstance(transformed, torch.Tensor):
        print(f"       typ: {type(transformed)}, shape: {tuple(transformed.shape)}, dtype: {transformed.dtype}")
        print(f"       urządzenie: {transformed.device}")
    else:
        print(f"       typ: {type(transformed)} (nie-tensor)")

    # --- 3️⃣ Inferencja (metryczna) ---
    pred = model.infer(transformed, f_px=f_px)
    depth = pred["depth"]

    print(f"[MODEL] Wyjście:")
    print(f"       typ: {type(depth)}, shape: {tuple(depth.shape)}, dtype: {depth.dtype}")
    if isinstance(depth, torch.Tensor):
        print(f"       urządzenie: {depth.device}")
        print(f"       zakres wartości: min={depth.min():.4f}, max={depth.max():.4f}")
    else:
        print(f"       zakres wartości: min={np.min(depth):.4f}, max={np.max(depth):.4f}")

    # --- 4️⃣ Konwersja i zapis ---
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = np.array(depth)

    depth_norm = (depth_np / np.max(depth_np) * 255).astype(np.uint8)
    out_path = os.path.join(folder, "depth_pro.png")
    cv2.imwrite(out_path, depth_norm)
    print(f"[SAVE] Zapisano mapę głębokości → {out_path}")
    print(f"       (depth_pro.png: uint8, zakres 0–255)")

print("\n[DONE] Wszystkie obrazy przetworzone.")
