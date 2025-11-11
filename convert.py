
import sys
import os
import torch
import glob

# --- konfiguracja ścieżek ---
sys.path.append("/content/VistaDreamDepthAnythingV2/ops")
from utils import save_ply

def convert_all_scenes(base_dir):
    """
    Wyszukuje wszystkie pliki scene.pth w podkatalogach base_dir
    i konwertuje każdy na scene.ply obok oryginału.
    """
    pth_files = glob.glob(os.path.join(base_dir, "**/scene.pth"), recursive=True)
    print(f"🔍 Znaleziono {len(pth_files)} plików scene.pth\n")

    for pth_path in sorted(pth_files):
        ply_path = os.path.join(os.path.dirname(pth_path), "scene.ply")
        print(f"🔄 Konwertuję: {pth_path}")
        try:
            scene = torch.load(pth_path, map_location="cpu")
            save_ply(scene, ply_path)
            print(f"✅ Zapisano: {ply_path}\n")
        except Exception as e:
            print(f"❌ Błąd przy {pth_path}: {e}\n")


if __name__ == "__main__":
    base_dir = "/content/drive/MyDrive/Colab Notebooks/GAUSIAN_SPLATTING_3D"
    convert_all_scenes(base_dir)
