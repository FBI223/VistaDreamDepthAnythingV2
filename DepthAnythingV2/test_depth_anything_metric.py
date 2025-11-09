import cv2, torch, numpy as np
#from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

import os
os.environ["XFORMERS_DISABLED"] = "1"

def detect_scene_type(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return "outdoor" if mean > 140 else "indoor"


# --- konfiguracja modelu ---
encoder = "vitl"          # LARGE model (bo vitl = Large)
dataset = "hypersim"      # indoor metric
max_depth = 20.0          # zakres metryczny dla hypersim



cfgs = {
    "vits": {"encoder":"vits","features":64,"out_channels":[48,96,192,384]},
    "vitb": {"encoder":"vitb","features":128,"out_channels":[96,192,384,768]},
    "vitl": {"encoder":"vitl","features":256,"out_channels":[256,512,1024,1024]}
}

model = DepthAnythingV2(**{**cfgs[encoder], "max_depth": max_depth})
ckpt = f"checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"
print("Loading:", ckpt)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
#model = model.to("cuda").eval()
model = model.to("cpu").eval()



#img = cv2.imread("input.jpg")      # BGR
#depth = model.infer_image(img)            # [metry]

#print("Depth range:", depth.min(), "→", depth.max(), "m")
#cv2.imwrite("depth_map.png", (depth / depth.max() * 255).astype(np.uint8))

# --- ścieżka bazowa ---
base_dir = os.path.dirname(__file__)
print("BASE  DIR:", base_dir)

# --- iteracja po folderach 1 ... 5 ---
for i in range(1, 7):
    folder = os.path.join("WAWEL_MODIFIED", str(i))
    img_path = os.path.join(folder, f"{i}.jpg" )

    if not os.path.exists(img_path):
        print(f"[WARN] Pomijam brakujący plik: {img_path}")
        continue

    print(f"[INFO] Przetwarzam: {img_path}")

    img = cv2.imread(img_path)  # BGR
    depth = model.infer_image(img)
    print(f"   Depth range: {depth.min():.3f} → {depth.max():.3f} m")

    depth_norm = (depth / depth.max() * 255).astype(np.uint8)
    out_path = os.path.join(folder, "depth_map.png")
    cv2.imwrite(out_path, depth_norm)
    print(f"   Zapisano: {out_path}")


print("[DONE] Wszystkie foldery przetworzone.")
