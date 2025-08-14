
import os
import io
import base64
import argparse
import yaml
import requests
from PIL import Image


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def img_to_b64(path):
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_a1111(cfg, a_pass_image, face_mask_path, contour_map_path, out_path):
    ep = cfg["general"]["a1111_endpoint"]
    bp = cfg["b_pass"]

    init_b64 = img_to_b64(a_pass_image)
    mask_b64 = img_to_b64(face_mask_path)
    contour_b64 = img_to_b64(contour_map_path) if os.path.exists(contour_map_path) else init_b64

    # Safe dimensions (<= 768 and multiples of 64)
    try:
        with Image.open(a_pass_image) as size_probe:
            src_w, src_h = size_probe.size
    except Exception:
        src_w, src_h = (768, 768)

    def to_multiple_of_64(value: float) -> int:
        return max(256, (int(value) // 64) * 64)

    max_side = 768
    scale = min(1.0, float(max_side) / float(max(src_w, src_h) or 1))
    tgt_w = to_multiple_of_64(src_w * scale)
    tgt_h = to_multiple_of_64(src_h * scale)

    payload = {
        "init_images": [init_b64],
        "mask": mask_b64,
        "prompt": bp.get("prompt", ""),
        "negative_prompt": bp.get("negative", ""),
        "denoising_strength": float(bp["denoise"]),
        "cfg_scale": float(bp["cfg"]),
        "steps": int(min(16, int(bp["steps"]))),
        "sampler_name": bp.get("sampler", "DPM++ SDE Karras"),
        "width": int(tgt_w),
        "height": int(tgt_h),
        "inpainting_fill": 1,
        "inpaint_only_masked": True,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "inpainting_mask_invert": 0,
        "mask_blur": int(bp["mask_blur_px"]),
        "override_settings": {
            **({"sd_model_checkpoint": cfg["general"].get("model_checkpoint")} if cfg["general"].get("model_checkpoint") else {})
        },
        "override_settings_restore_afterwards": True,
        "alwayson_scripts": {}
    }

    # ADetailer configuration
    ad_args = []
    if bp.get("use_adetailer", False):
        ad_args.append({
            "ad_model": bp.get("ad_model", "face_yolov8n.pt"),
            "ad_confidence": float(bp.get("ad_confidence", 0.33)),
            "ad_mask_min_ratio": float(bp.get("ad_mask_min_ratio", 0.01)),
            "ad_dilate_erode": int(bp.get("ad_dilate_erode", 16)),
            "ad_mask_blur": int(bp.get("ad_mask_blur", 14)),
            "ad_inpaint_only_masked": True,
            "ad_denoising_strength": float(bp.get("ad_denoise", bp["denoise"])),
            "ad_use_inpaint": True,
            "ad_use_next_frame": False,
            "ad_prompt": bp.get("ad_prompt", ""),
            "ad_negative_prompt": bp.get("ad_negative", "")
        })

    if bp.get("use_adetailer2", False):
        ad_args.append({
            "ad_model": bp.get("ad2_model", bp.get("ad_model", "face_yolov8n.pt")),
            "ad_confidence": float(bp.get("ad2_confidence", bp.get("ad_confidence", 0.33))),
            "ad_mask_min_ratio": float(bp.get("ad2_mask_min_ratio", bp.get("ad_mask_min_ratio", 0.01))),
            "ad_dilate_erode": int(bp.get("ad2_dilate_erode", bp.get("ad_dilate_erode", 16))),
            "ad_mask_blur": int(bp.get("ad2_mask_blur", bp.get("ad_mask_blur", 14))),
            "ad_inpaint_only_masked": True,
            "ad_denoising_strength": float(bp.get("ad2_denoise", bp.get("ad_denoise", bp["denoise"]))),
            "ad_use_inpaint": True,
            "ad_use_next_frame": False,
            "ad_prompt": bp.get("ad2_prompt", bp.get("ad_prompt", "")),
            "ad_negative_prompt": bp.get("ad2_negative", bp.get("ad_negative", ""))
        })

    if ad_args:
        payload["alwayson_scripts"]["ADetailer"] = {"args": ad_args}

    # ControlNet configuration
    cn_args = []
    def map_control_mode(value):
        mapping = {
            0: "Balanced",
            1: "My prompt is more important",
            2: "ControlNet is more important",
        }
        try:
            iv = int(value)
        except Exception:
            return "Balanced"
        return mapping.get(iv, "Balanced")

    if bp.get("use_controlnet2", False):
        cn2_model = bp.get("controlnet2_model") or "xinsirControlnetCanny_v20"
        cn2_image_b64 = f"data:image/png;base64,{contour_b64}"
        cn_args.append({
            "enabled": True,
            "module": bp.get("controlnet2_module", "canny"),
            "model": cn2_model,
            "weight": float(bp.get("controlnet2_weight", 0.44)),
            "input_image": cn2_image_b64,
            "lowvram": False,
            "processor_res": 512,
            "guidance_start": float(bp.get("controlnet2_guidance_start", 0.25)),
            "guidance_end": float(bp.get("controlnet2_guidance_end", 0.90)),
            "control_mode": map_control_mode(bp.get("controlnet2_mode", 0)),
            "threshold_a": 100,
            "threshold_b": 200,
        })

    if cn_args:
        payload["alwayson_scripts"]["ControlNet"] = {"args": cn_args}

    r = requests.post(f"{ep}/sdapi/v1/img2img", json=payload, timeout=900)
    if r.status_code != 200:
        raise RuntimeError(f"img2img failed HTTP {r.status_code}: {r.text[:1000]}")
    
    data = r.json()
    if "images" not in data or not data["images"]:
        raise RuntimeError("No images from A1111")
    
    img_b64 = data["images"][0].split(",", 1)[-1]
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(img_b64))
    print(f"[B-PASS] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--output", default="output")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    a_pass_image = os.path.join(args.workdir, "a_pass", "base_enhanced.png")
    face_mask_path = os.path.join(args.workdir, "masks", "face_mask.png")
    contour_map_path = os.path.join(args.workdir, "a_pass", "contour_map.png")
    os.makedirs(args.output, exist_ok=True)
    
    input_name = os.path.basename(os.path.normpath(args.workdir)) or "final"
    out_path = os.path.join(args.output, f"{input_name}.png")

    run_a1111(cfg, a_pass_image, face_mask_path, contour_map_path, out_path)


if __name__ == "__main__":
    main()
