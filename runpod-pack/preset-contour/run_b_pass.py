
import os
import io
import base64
import argparse
import webbrowser
import yaml
import requests
from PIL import Image, ImageChops, ImageStat, ImageDraw, ImageFilter
import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def img_to_b64(path):
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def quick_face_check_for_sequential_mode(image_path: str) -> dict:
    """Швидка евристика на базі OpenCV: рахуємо детекції при різних поворотах.
    Повертаємо словник з лічильниками для 0, 90, 180, 270 градусів.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"0": 0, "90": 0, "180": 0, "270": 0}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Використовуємо стандартний каскад Хаара (як базову евристику)
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        def count_faces(mat):
            faces = haar.detectMultiScale(mat, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))
            return len(faces) if faces is not None else 0
        counts = {"0": count_faces(gray)}
        # 90
        counts["90"] = count_faces(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE))
        # 180
        counts["180"] = count_faces(cv2.rotate(gray, cv2.ROTATE_180))
        # 270
        counts["270"] = count_faces(cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE))
        return counts
    except Exception:
        return {"0": 0, "90": 0, "180": 0, "270": 0}


def choose_single_adetailer_pass(bp: dict, image_path: str) -> list:
    """Вибираємо лише один набір аргументів ADetailer залежно від швидкої перевірки облич.
    Порядок пріоритетів: ad → ad2 → ad3. Якщо нічого не виявлено в "0", але є в поворотах — віддаємо пріоритет ad3.
    Повертаємо список з 0 або 1 елементом (для payload)."""
    if not bp.get("use_sequential_adetailer", False):
        return []
    counts = quick_face_check_for_sequential_mode(image_path)
    # Якщо є обличчя без ротації, спробуємо йти від слабшого до сильнішого
    if max(counts.get("0", 0), 0) > 0:
        if bp.get("use_adetailer", False):
            return [{
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
            }]
        elif bp.get("use_adetailer2", False):
            return [{
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
            }]
        elif bp.get("use_adetailer3", False):
            return [{
                "ad_model": bp.get("ad3_model", bp.get("ad2_model", bp.get("ad_model", "face_yolov8n.pt"))),
                "ad_confidence": float(bp.get("ad3_confidence", bp.get("ad2_confidence", bp.get("ad_confidence", 0.33)))),
                "ad_mask_min_ratio": float(bp.get("ad3_mask_min_ratio", bp.get("ad2_mask_min_ratio", bp.get("ad_mask_min_ratio", 0.01)))),
                "ad_dilate_erode": int(bp.get("ad3_dilate_erode", bp.get("ad2_dilate_erode", bp.get("ad_dilate_erode", 16)))),
                "ad_mask_blur": int(bp.get("ad3_mask_blur", bp.get("ad2_mask_blur", bp.get("ad_mask_blur", 14)))),
                "ad_inpaint_only_masked": True,
                "ad_denoising_strength": float(bp.get("ad3_denoise", bp.get("ad2_denoise", bp.get("ad_denoise", bp["denoise"])))),
                "ad_use_inpaint": True,
                "ad_use_next_frame": False,
                "ad_prompt": bp.get("ad3_prompt", bp.get("ad2_prompt", bp.get("ad_prompt", ""))),
                "ad_negative_prompt": bp.get("ad3_negative", bp.get("ad2_negative", bp.get("ad_negative", "")))
            }]
    # Якщо немає у 0°, але є у поворотах — віддаємо перевагу найсильнішому/поблажливому ad3
    if max(counts.get("90", 0), counts.get("180", 0), counts.get("270", 0)) > 0:
        if bp.get("use_adetailer3", False):
            return [{
                "ad_model": bp.get("ad3_model", bp.get("ad2_model", bp.get("ad_model", "face_yolov8n.pt"))),
                "ad_confidence": float(bp.get("ad3_confidence", bp.get("ad2_confidence", bp.get("ad_confidence", 0.33)))),
                "ad_mask_min_ratio": float(bp.get("ad3_mask_min_ratio", bp.get("ad2_mask_min_ratio", bp.get("ad_mask_min_ratio", 0.01)))),
                "ad_dilate_erode": int(bp.get("ad3_dilate_erode", bp.get("ad2_dilate_erode", bp.get("ad_dilate_erode", 16)))),
                "ad_mask_blur": int(bp.get("ad3_mask_blur", bp.get("ad2_mask_blur", bp.get("ad_mask_blur", 14)))),
                "ad_inpaint_only_masked": True,
                "ad_denoising_strength": float(bp.get("ad3_denoise", bp.get("ad2_denoise", bp.get("ad_denoise", bp["denoise"])))),
                "ad_use_inpaint": True,
                "ad_use_next_frame": False,
                "ad_prompt": bp.get("ad3_prompt", bp.get("ad2_prompt", bp.get("ad_prompt", ""))),
                "ad_negative_prompt": bp.get("ad3_negative", bp.get("ad2_negative", bp.get("ad_negative", "")))
            }]
        # fallback якщо ad3 вимкнений
        for key in ("use_adetailer2", "use_adetailer"):
            if bp.get(key, False):
                return choose_single_adetailer_pass({**bp, "use_sequential_adetailer": True, key: True}, image_path)
    # Якщо взагалі нічого, повертаємо порожній список — тобто ADetailer не запускати
    return []


def _encode_blank_mask_like(image_path: str) -> str:
    """Створює повністю чорну маску розміру зображення та повертає base64 (PNG)."""
    with Image.open(image_path) as im:
        blank = Image.new("L", im.size, 0)
        buf = io.BytesIO()
        blank.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_ad_args_for_variant(bp: dict, variant: str) -> dict:
    if variant == "ad":
        return {
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
        }
    if variant == "ad2":
        return {
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
        }
    # ad3
    return {
        "ad_model": bp.get("ad3_model", bp.get("ad2_model", bp.get("ad_model", "face_yolov8n.pt"))),
        "ad_confidence": float(bp.get("ad3_confidence", bp.get("ad2_confidence", bp.get("ad_confidence", 0.33)))),
        "ad_mask_min_ratio": float(bp.get("ad3_mask_min_ratio", bp.get("ad2_mask_min_ratio", bp.get("ad_mask_min_ratio", 0.01)))),
        "ad_dilate_erode": int(bp.get("ad3_dilate_erode", bp.get("ad2_dilate_erode", bp.get("ad_dilate_erode", 16)))),
        "ad_mask_blur": int(bp.get("ad3_mask_blur", bp.get("ad2_mask_blur", bp.get("ad_mask_blur", 14)))),
        "ad_inpaint_only_masked": True,
        "ad_denoising_strength": float(bp.get("ad3_denoise", bp.get("ad2_denoise", bp.get("ad_denoise", bp["denoise"])))),
        "ad_use_inpaint": True,
        "ad_use_next_frame": False,
        "ad_prompt": bp.get("ad3_prompt", bp.get("ad2_prompt", bp.get("ad_prompt", ""))),
        "ad_negative_prompt": bp.get("ad3_negative", bp.get("ad2_negative", bp.get("ad_negative", "")))
    }


def _images_different(img_path_a: str, img_bytes_b: bytes, mse_threshold: float = 1.0) -> bool:
    """Порівнюємо вхідне зображення і результат. Якщо MSE > threshold — вважаємо, що були зміни."""
    with Image.open(img_path_a).convert("RGB") as im_a:
        with Image.open(io.BytesIO(img_bytes_b)).convert("RGB") as im_b:
            if im_a.size != im_b.size:
                im_b = im_b.resize(im_a.size)
            diff = ImageChops.difference(im_a, im_b)
            stat = ImageStat.Stat(diff)
            # Середньоквадратичне по каналах
            mse = sum((v ** 2 for v in stat.mean)) / len(stat.mean)
            return mse > mse_threshold


def _try_adetailer_probe(ep: str, cfg_general: dict, bp: dict, a_pass_image: str, variant: str) -> bool:
    """Запуск короткої «проби» лише з ADetailer (без ControlNet і без базової маски) та перевірка, чи є зміни."""
    init_b64 = img_to_b64(a_pass_image)
    blank_mask_b64 = _encode_blank_mask_like(a_pass_image)

    # Розмір як у основному пайплайні
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

    ad_args = [_build_ad_args_for_variant(bp, variant)]

    payload = {
        "init_images": [init_b64],
        "mask": blank_mask_b64,
        "prompt": "",
        "negative_prompt": "",
        "denoising_strength": float(bp.get("denoise", 0.18)),
        "cfg_scale": float(bp.get("cfg", 5.0)),
        "steps": int(min(16, int(bp.get("steps", 32)))) ,
        "sampler_name": bp.get("sampler", "DPM++ SDE Karras"),
        "width": int(tgt_w),
        "height": int(tgt_h),
        "inpainting_fill": 1,
        "inpaint_only_masked": True,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "inpainting_mask_invert": 0,
        "mask_blur": int(bp.get("mask_blur_px", 6)),
        "override_settings": {
            **({"sd_model_checkpoint": cfg_general.get("model_checkpoint")} if cfg_general.get("model_checkpoint") else {})
        },
        "override_settings_restore_afterwards": True,
        "alwayson_scripts": {
            "ADetailer": {"args": ad_args}
        }
    }

    r = requests.post(f"{ep}/sdapi/v1/img2img", json=payload, timeout=900)
    if r.status_code != 200:
        return False
    data = r.json()
    if "images" not in data or not data["images"]:
        return False
    img_b64 = data["images"][0].split(",", 1)[-1]
    out_bytes = base64.b64decode(img_b64)
    return _images_different(a_pass_image, out_bytes)


def _generate_cheek_mask_facemesh(image_path: str, dilate_px: int = 10, blur_px: int = 12) -> Image.Image:
    """Генеруємо маску щік за допомогою MediaPipe FaceMesh (приблизний полігон малярних зон).
    Повертає PIL L-маску. Якщо невдача — повертає None.
    """
    if not _HAS_MP:
        return None
    try:
        with Image.open(image_path) as im:
            rgb = np.array(im.convert("RGB"))
        h, w = rgb.shape[:2]
        mp_faces = mp.solutions.face_mesh
        with mp_faces.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
            res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        # Евристичний полігон щік: з'єднуємо області біля скули (блок індексів можна уточнювати)
        # Використаємо частини ліній щік з орієнтиром: вушка/щелепа (індекси 93..132 приблизно у MP), спростимо масив
        left_cheek_idx = [50, 101, 118, 117, 123, 147, 177, 215, 50]
        right_cheek_idx = [280, 330, 347, 346, 352, 376, 405, 435, 280]
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        def draw_poly(indexes):
            poly = [pts[i] for i in indexes if 0 <= i < len(pts)]
            if len(poly) >= 3:
                draw.polygon(poly, fill=255)
        draw_poly(left_cheek_idx)
        draw_poly(right_cheek_idx)
        # Dilate/blur
        if dilate_px > 0:
            mask = mask.filter(ImageFilter.MaxFilter(size=max(3, dilate_px | 1)))
        if blur_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_px))
        return mask
    except Exception:
        return None


def _img_bytes_from_b64img(b64img: str) -> bytes:
    return base64.b64decode(b64img.split(",", 1)[-1])


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
    if bp.get("use_true_fallback_adetailer", False):
        # Послідовно пробуємо ad → ad2 → ad3 і зупиняємось на першому, що реально дав зміни
        for variant_key, enabled_flag in (("ad", "use_adetailer"), ("ad2", "use_adetailer2"), ("ad3", "use_adetailer3")):
            if bp.get(enabled_flag, False):
                try:
                    if _try_adetailer_probe(ep, cfg["general"], bp, a_pass_image, variant_key):
                        ad_args = [_build_ad_args_for_variant(bp, variant_key)]
                        break
                except Exception:
                    continue
    elif bp.get("use_sequential_adetailer", False):
        ad_args = choose_single_adetailer_pass(bp, a_pass_image)
    else:
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
        if bp.get("use_adetailer3", False):
            ad_args.append({
                "ad_model": bp.get("ad3_model", bp.get("ad2_model", bp.get("ad_model", "face_yolov8n.pt"))),
                "ad_confidence": float(bp.get("ad3_confidence", bp.get("ad2_confidence", bp.get("ad_confidence", 0.33)))),
                "ad_mask_min_ratio": float(bp.get("ad3_mask_min_ratio", bp.get("ad2_mask_min_ratio", bp.get("ad_mask_min_ratio", 0.01)))),
                "ad_dilate_erode": int(bp.get("ad3_dilate_erode", bp.get("ad2_dilate_erode", bp.get("ad_dilate_erode", 16)))),
                "ad_mask_blur": int(bp.get("ad3_mask_blur", bp.get("ad2_mask_blur", bp.get("ad_mask_blur", 14)))),
                "ad_inpaint_only_masked": True,
                "ad_denoising_strength": float(bp.get("ad3_denoise", bp.get("ad2_denoise", bp.get("ad_denoise", bp["denoise"])))),
                "ad_use_inpaint": True,
                "ad_use_next_frame": False,
                "ad_prompt": bp.get("ad3_prompt", bp.get("ad2_prompt", bp.get("ad_prompt", ""))),
                "ad_negative_prompt": bp.get("ad3_negative", bp.get("ad2_negative", bp.get("ad_negative", "")))
            })

    if ad_args:
        payload["alwayson_scripts"]["ADetailer"] = {"args": ad_args}
    else:
        # Якщо увімкнений ручний фолбек — дозволяємо користувачу намалювати маску і/або відкрити UI
        if bp.get("enable_manual_fallback", False):
            manual_mask_name = bp.get("manual_mask_filename", "custom_face_mask.png")
            manual_mask_path = os.path.join(os.path.dirname(face_mask_path), manual_mask_name)
            if os.path.exists(manual_mask_path):
                try:
                    mask_b64 = img_to_b64(manual_mask_path)
                    payload["mask"] = mask_b64
                    # Мінімальна інпейнт-логіка без ADetailer: використаємо ту ж denoise/steps
                except Exception:
                    pass
            if bp.get("open_ui_on_manual", False) and ep:
                try:
                    webbrowser.open(ep)
                except Exception:
                    pass
            if bp.get("open_manual_help_on_manual", False):
                help_path = bp.get("manual_help_path", "manual_fallback_help.html")
                try:
                    if not os.path.isabs(help_path):
                        # Відносно поточного файлу
                        base_dir = os.path.dirname(__file__)
                        help_path = os.path.join(base_dir, help_path)
                    if os.path.exists(help_path):
                        webbrowser.open(f"file://{help_path}")
                except Exception:
                    pass

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

    # Optional Cheek Refiner pass
    bp = cfg["b_pass"]
    if bp.get("use_cheek_refiner", False):
        cheek_mask_path = os.path.join(os.path.dirname(face_mask_path), "cheek_mask.png")
        cheek_mask = None
        if bp.get("cheek_mask_source", "facemesh") == "facemesh":
            cheek_mask = _generate_cheek_mask_facemesh(
                out_path,
                dilate_px=int(bp.get("cheek_mask_dilate", 10)),
                blur_px=int(bp.get("cheek_mask_blur", 12))
            )
        if cheek_mask is None and os.path.exists(cheek_mask_path):
            try:
                cheek_mask = Image.open(cheek_mask_path).convert("L")
            except Exception:
                cheek_mask = None
        if cheek_mask is not None:
            try:
                cheek_mask.save(cheek_mask_path)
                init_b64 = img_to_b64(out_path)
                mask_b64 = img_to_b64(cheek_mask_path)
                # Розмір залишаємо той же, що у out_path
                with Image.open(out_path) as size_probe2:
                    src_w2, src_h2 = size_probe2.size
                def to_multiple_of_64(value: float) -> int:
                    return max(256, (int(value) // 64) * 64)
                max_side2 = 768
                scale2 = min(1.0, float(max_side2) / float(max(src_w2, src_h2) or 1))
                tgt_w2 = to_multiple_of_64(src_w2 * scale2)
                tgt_h2 = to_multiple_of_64(src_h2 * scale2)
                cheek_payload = {
                    "init_images": [init_b64],
                    "mask": mask_b64,
                    "prompt": bp.get("cheek_prompt", ""),
                    "negative_prompt": bp.get("cheek_negative", ""),
                    "denoising_strength": float(bp.get("cheek_denoise", 0.16)),
                    "cfg_scale": float(bp.get("cfg", 5.0)),
                    "steps": int(min(16, int(bp.get("steps", 32)))),
                    "sampler_name": bp.get("sampler", "DPM++ SDE Karras"),
                    "width": int(tgt_w2),
                    "height": int(tgt_h2),
                    "inpainting_fill": 1,
                    "inpaint_only_masked": True,
                    "inpaint_full_res": True,
                    "inpaint_full_res_padding": 32,
                    "inpainting_mask_invert": 0,
                    "mask_blur": int(bp.get("mask_blur_px", 6)),
                    "override_settings": {
                        **({"sd_model_checkpoint": cfg["general"].get("model_checkpoint")} if cfg["general"].get("model_checkpoint") else {})
                    },
                    "override_settings_restore_afterwards": True,
                    "alwayson_scripts": {}
                }
                r2 = requests.post(f"{ep}/sdapi/v1/img2img", json=cheek_payload, timeout=900)
                if r2.status_code == 200:
                    data2 = r2.json()
                    if "images" in data2 and data2["images"]:
                        img2_b64 = data2["images"][0].split(",", 1)[-1]
                        with open(out_path, "wb") as f:
                            f.write(base64.b64decode(img2_b64))
                        print(f"[CHEEK-REFINER] Saved: {out_path}")
            except Exception:
                pass

    # Optional Eyelash Refiner pass
    if bp.get("use_eyelash_refiner", False):
        eyelash_mask_path = os.path.join(os.path.dirname(face_mask_path), "eyelash_mask.png")
        eyelash_mask = None
        if bp.get("eyelash_mask_source", "facemesh") == "facemesh" and _HAS_MP:
            try:
                with Image.open(out_path) as im_eye:
                    rgb = np.array(im_eye.convert("RGB"))
                h3, w3 = rgb.shape[:2]
                mp_faces = mp.solutions.face_mesh
                with mp_faces.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm3:
                    res3 = fm3.process(rgb)
                if res3.multi_face_landmarks:
                    lm3 = res3.multi_face_landmarks[0]
                    pts3 = [(int(p.x * w3), int(p.y * h3)) for p in lm3.landmark]
                    # Окреслення очей (контури) — списки індексів MediaPipe (поширені у прикладах)
                    right_eye_contour = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    left_eye_contour = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    m_eye = Image.new("L", (w3, h3), 0)
                    d_eye = ImageDraw.Draw(m_eye)
                    def poly_from(idxs):
                        poly = [pts3[i] for i in idxs if 0 <= i < len(pts3)]
                        if len(poly) >= 3:
                            d_eye.polygon(poly, fill=255)
                    poly_from(right_eye_contour)
                    poly_from(left_eye_contour)
                    # Злегка звузимо до верхньої половини очей, щоб сфокусуватись на віях
                    upper_half = Image.new("L", (w3, h3), 0)
                    d_up = ImageDraw.Draw(upper_half)
                    d_up.rectangle([0, 0, w3, int(h3 * 0.5)], fill=255)
                    m_eye = ImageChops.multiply(m_eye, upper_half)
                    # Dilate/blur за конфігом
                    dil = int(bp.get("eyelash_mask_dilate", 4))
                    blr = int(bp.get("eyelash_mask_blur", 6))
                    if dil > 0:
                        m_eye = m_eye.filter(ImageFilter.MaxFilter(size=max(3, dil | 1)))
                    if blr > 0:
                        m_eye = m_eye.filter(ImageFilter.GaussianBlur(radius=blr))
                    eyelash_mask = m_eye
            except Exception:
                eyelash_mask = None
        if eyelash_mask is None and os.path.exists(eyelash_mask_path):
            try:
                eyelash_mask = Image.open(eyelash_mask_path).convert("L")
            except Exception:
                eyelash_mask = None
        if eyelash_mask is not None:
            try:
                eyelash_mask.save(eyelash_mask_path)
                init_b64_ey = img_to_b64(out_path)
                mask_b64_ey = img_to_b64(eyelash_mask_path)
                with Image.open(out_path) as size_probe3:
                    src_w3, src_h3 = size_probe3.size
                def to_multiple_of_64(value: float) -> int:
                    return max(256, (int(value) // 64) * 64)
                max_side3 = 768
                scale3 = min(1.0, float(max_side3) / float(max(src_w3, src_h3) or 1))
                tgt_w3 = to_multiple_of_64(src_w3 * scale3)
                tgt_h3 = to_multiple_of_64(src_h3 * scale3)
                eyelash_payload = {
                    "init_images": [init_b64_ey],
                    "mask": mask_b64_ey,
                    "prompt": bp.get("eyelash_prompt", ""),
                    "negative_prompt": bp.get("eyelash_negative", ""),
                    "denoising_strength": float(bp.get("eyelash_denoise", 0.14)),
                    "cfg_scale": float(bp.get("cfg", 5.0)),
                    "steps": int(min(16, int(bp.get("steps", 32)))),
                    "sampler_name": bp.get("sampler", "DPM++ SDE Karras"),
                    "width": int(tgt_w3),
                    "height": int(tgt_h3),
                    "inpainting_fill": 1,
                    "inpaint_only_masked": True,
                    "inpaint_full_res": True,
                    "inpaint_full_res_padding": 32,
                    "inpainting_mask_invert": 0,
                    "mask_blur": int(bp.get("mask_blur_px", 6)),
                    "override_settings": {
                        **({"sd_model_checkpoint": cfg["general"].get("model_checkpoint")} if cfg["general"].get("model_checkpoint") else {})
                    },
                    "override_settings_restore_afterwards": True,
                    "alwayson_scripts": {}
                }
                r3 = requests.post(f"{ep}/sdapi/v1/img2img", json=eyelash_payload, timeout=900)
                if r3.status_code == 200:
                    data3 = r3.json()
                    if "images" in data3 and data3["images"]:
                        img3_b64 = data3["images"][0].split(",", 1)[-1]
                        with open(out_path, "wb") as f:
                            f.write(base64.b64decode(img3_b64))
                        print(f"[EYELASH-REFINER] Saved: {out_path}")
            except Exception:
                pass


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
