#!/usr/bin/env bash
set -e

echo "📦 Tworzenie folderów..."
mkdir -p tools/Fooocus/models/checkpoints
mkdir -p tools/Fooocus/models/loras
mkdir -p tools/Fooocus/models/inpaint
mkdir -p tools/Fooocus/models/prompt_expansion/fooocus_expansion
mkdir -p tools/DepthPro/checkpoints
mkdir -p tools/OneFormer/checkpoints
mkdir -p tools/StableDiffusion/lcm_ckpt
mkdir -p DepthAnythingV2/checkpoints

echo "⬇️  Pobieranie modelu Fooocus (podstawowy)..."
wget -nc -O tools/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors \
  https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors

echo "⬇️  Pobieranie modelu Fooocus LoRA..."
wget -nc -O tools/Fooocus/models/loras/sd_xl_offset_example-lora_1.0.safetensors \
  https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors

echo "⬇️  Pobieranie modelu Fooocus Inpaint..."
wget -nc -O tools/Fooocus/models/inpaint/inpaint_v26.fooocus.patch \
  https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch?download=true

echo "⬇️  Pobieranie Fooocus Prompt-Extension..."
wget -nc -O tools/Fooocus/models/prompt_expansion/fooocus_expansion/pytorch_model.bin \
  https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin?download=true

echo "⬇️  Pobieranie modelu DepthPro..."
wget -nc -O tools/DepthPro/checkpoints/depth_pro.pt \
  https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

echo "⬇️  Pobieranie modelu OneFormer ADE20K..."
wget -nc -O tools/OneFormer/checkpoints/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth \
  https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth

echo "⬇️  Pobieranie modelu Stable Diffusion LCM..."
wget -nc -O tools/StableDiffusion/lcm_ckpt/pytorch_lora_weights.safetensors \
  https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors

echo "⬇️  Pobieranie Depth Anything V2 (LARGE)..."
wget -nc -O DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

echo "✅ Wszystkie modele pobrane poprawnie!"