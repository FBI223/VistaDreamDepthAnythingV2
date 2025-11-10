
import os
from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline


# Ścieżka bazowa do folderów WAWEL
BASE_DIR = "VistaDreamDepthAnythingV2/WAWEL_MODIFIED"

# Wczytanie konfiguracji
cfg = load_cfg("pipe/cfgs/basic.yaml")

# Przejście przez foldery 1–6
for i in range(1, 7):
    folder_path = os.path.join(BASE_DIR, str(i))
    image_path = os.path.join(folder_path, f"{i}.jpg")

    if not os.path.exists(image_path):
        print(f"❌ Brak pliku: {image_path}")
        continue

    print(f"\n=== 🔷 Przetwarzanie folderu {i} → {image_path} ===")
    cfg.scene.input.rgb = image_path  # ustaw aktualny obraz
    vistadream = Pipeline(cfg)        # twórz pipeline dla każdej sceny
    vistadream()                      # uruchom przetwarzanie
    print(f"✅ Zakończono przetwarzanie {i}.jpg\n")