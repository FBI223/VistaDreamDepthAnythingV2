import os
from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline

# 📂 Bazowy folder (zmień jeśli inna lokalizacja)
BASE_DIR = "/content/VistaDreamDepthAnythingV2/DepthAnythingV2/WAWEL_MODIFIED"

# ⚙️ Wczytaj konfigurację
cfg = load_cfg("pipe/cfgs/basic.yaml")

# 🔁 Iteruj po folderach 3-3
for i in range(3, 4):
    folder_path = os.path.join(BASE_DIR, str(i))
    image_path = os.path.join(folder_path, f"{i}.jpg")

    if not os.path.exists(image_path):
        print(f"❌ Brak pliku: {image_path}")
        continue

    # 📤 Folder wyjściowy
    output_dir = os.path.join(folder_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== 🔷 Przetwarzanie folderu {i} → {image_path} ===")

    # Ustaw wejście i wyjście w konfiguracji
    cfg.scene.input.rgb = image_path
    cfg.scene.output.dir = output_dir

    # Uruchom pipeline
    vistadream = Pipeline(cfg)
    vistadream()

    print(f"✅ Wyniki zapisane w: {output_dir}\n")