
import os
import argparse
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps


def tight_face_mask(img):
    w, h = img.size
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    cx, cy = int(w * 0.50), int(h * 0.48)
    rx, ry = int(w * 0.22), int(h * 0.28)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    
    # Remove eyes/mouth/top/neck
    eye_rx, eye_ry = int(w * 0.088), int(h * 0.046)
    for ex in (int(w * 0.34), int(w * 0.66)):
        ImageDraw.Draw(m).ellipse([ex - eye_rx, int(h * 0.41) - eye_ry, ex + eye_rx, int(h * 0.41) + eye_ry], fill=0)
    
    ImageDraw.Draw(m).ellipse([int(w * 0.50) - int(w * 0.15), int(h * 0.62) - int(h * 0.050),
                               int(w * 0.50) + int(w * 0.15), int(h * 0.62) + int(h * 0.050)], fill=0)
    ImageDraw.Draw(m).rectangle([0, 0, w, int(h * 0.18)], fill=0)
    ImageDraw.Draw(m).rectangle([0, int(h * 0.65), w, h], fill=0)
    
    m = m.filter(ImageFilter.GaussianBlur(radius=int(max(w, h) * 0.012)))
    
    # Shrink mask to limit edits strictly to the face
    shrink_px = max(3, int(min(w, h) * 0.012)) | 1
    try:
        m = m.filter(ImageFilter.MinFilter(size=shrink_px))
    except Exception:
        pass
    return m


def line_mask(w, h, points, width_rel=0.004, blur_rel=0.010):
    m = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(m)
    w_line = max(1, int(min(w, h) * width_rel))
    draw.line(points, fill=255, width=w_line, joint="curve")
    return m.filter(ImageFilter.GaussianBlur(radius=int(max(w, h) * blur_rel)))


def ellipse_stroke_mask(w, h, cx, cy, rx, ry, width_rel=0.0035, blur_rel=0.010):
    """Create a thin elliptical stroke by subtracting two ellipses."""
    outer = Image.new("L", (w, h), 0)
    inner = Image.new("L", (w, h), 0)
    dw = max(1, int(min(w, h) * width_rel))
    ImageDraw.Draw(outer).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    ImageDraw.Draw(inner).ellipse([cx - rx + dw, cy - ry + dw, cx + rx - dw, cy + ry - dw], fill=255)
    stroke = ImageChops.subtract(outer, inner)
    return stroke.filter(ImageFilter.GaussianBlur(radius=int(max(w, h) * blur_rel)))


def ellipse_mask(w, h, cx, cy, rx, ry, blur_rel):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return m.filter(ImageFilter.GaussianBlur(radius=int(max(w, h) * blur_rel)))


def union(*masks):
    out = Image.new("L", masks[0].size, 0)
    for mm in masks:
        out = ImageChops.lighter(out, mm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--workdir", default="work")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.workdir, "a_pass"), exist_ok=True)
    os.makedirs(os.path.join(args.workdir, "masks"), exist_ok=True)

    img = Image.open(args.input).convert("RGB")
    w, h = img.size

    # Face mask
    face = tight_face_mask(img)
    face_path = os.path.join(args.workdir, "masks", "face_mask.png")
    face.save(face_path)

    # Base image enhancement: unsharp only on face area
    base_path = os.path.join(args.workdir, "a_pass", "base_enhanced.png")
    img_sharp = img.filter(ImageFilter.UnsharpMask(radius=max(w, h) * 0.004, percent=180, threshold=2))
    base_enhanced = Image.composite(img_sharp, img, face)
    base_enhanced.save(base_path)

    # Contour map for ControlNet2
    white = Image.new("L", (w, h), 255)
    
    # Base: face oval stroke and jawline
    oval = ellipse_stroke_mask(w, h, int(w * 0.50), int(h * 0.50), int(w * 0.24), int(h * 0.30), width_rel=0.0032, blur_rel=0.010)
    jaw = line_mask(w, h, [(int(w * 0.38), int(h * 0.64)), (int(w * 0.50), int(h * 0.648)), (int(w * 0.62), int(h * 0.64))], width_rel=0.0032, blur_rel=0.012)

    # Temples / hairline short arcs
    temple_L = line_mask(w, h, [(int(w * 0.30), int(h * 0.28)), (int(w * 0.28), int(h * 0.24)), (int(w * 0.32), int(h * 0.22))], width_rel=0.003, blur_rel=0.012)
    temple_R = line_mask(w, h, [(int(w * 0.70), int(h * 0.28)), (int(w * 0.72), int(h * 0.24)), (int(w * 0.68), int(h * 0.22))], width_rel=0.003, blur_rel=0.012)

    # Cheekbone accent lines
    cheek_main_L = line_mask(w, h, [(int(w * 0.33), int(h * 0.56)), (int(w * 0.40), int(h * 0.59)), (int(w * 0.46), int(h * 0.58))], width_rel=0.0060, blur_rel=0.005)
    cheek_main_R = line_mask(w, h, [(int(w * 0.67), int(h * 0.56)), (int(w * 0.60), int(h * 0.59)), (int(w * 0.54), int(h * 0.58))], width_rel=0.0060, blur_rel=0.005)
    cheek_upper_L = line_mask(w, h, [(int(w * 0.36), int(h * 0.50)), (int(w * 0.42), int(h * 0.52)), (int(w * 0.48), int(h * 0.51))], width_rel=0.0042, blur_rel=0.005)
    cheek_upper_R = line_mask(w, h, [(int(w * 0.64), int(h * 0.50)), (int(w * 0.58), int(h * 0.52)), (int(w * 0.52), int(h * 0.51))], width_rel=0.0042, blur_rel=0.005)
    cheek_ridge_L = line_mask(w, h, [(int(w * 0.33), int(h * 0.54)), (int(w * 0.40), int(h * 0.565)), (int(w * 0.47), int(h * 0.555))], width_rel=0.0030, blur_rel=0.004)
    cheek_ridge_R = line_mask(w, h, [(int(w * 0.67), int(h * 0.54)), (int(w * 0.60), int(h * 0.565)), (int(w * 0.53), int(h * 0.555))], width_rel=0.0030, blur_rel=0.004)

    # Cheek shadow line toward mouth corners
    cheek_shadow_L = line_mask(w, h, [(int(w * 0.42), int(h * 0.60)), (int(w * 0.46), int(h * 0.62)), (int(w * 0.48), int(h * 0.64))], width_rel=0.0042, blur_rel=0.009)
    cheek_shadow_R = line_mask(w, h, [(int(w * 0.58), int(h * 0.60)), (int(w * 0.54), int(h * 0.62)), (int(w * 0.52), int(h * 0.64))], width_rel=0.0042, blur_rel=0.009)

    # Forehead soft edge
    forehead_edge = line_mask(w, h, [(int(w * 0.38), int(h * 0.26)), (int(w * 0.50), int(h * 0.24)), (int(w * 0.62), int(h * 0.26))], width_rel=0.0028, blur_rel=0.012)

    # Eyelids and crow's feet
    upper_lid_L = line_mask(w, h, [(int(w * 0.40), int(h * 0.41)), (int(w * 0.45), int(h * 0.405)), (int(w * 0.50), int(h * 0.41))], width_rel=0.0034, blur_rel=0.006)
    lower_lid_L = line_mask(w, h, [(int(w * 0.40), int(h * 0.43)), (int(w * 0.45), int(h * 0.435)), (int(w * 0.50), int(h * 0.43))], width_rel=0.0029, blur_rel=0.006)
    upper_lid_R = line_mask(w, h, [(int(w * 0.60), int(h * 0.41)), (int(w * 0.55), int(h * 0.405)), (int(w * 0.50), int(h * 0.41))], width_rel=0.0034, blur_rel=0.006)
    lower_lid_R = line_mask(w, h, [(int(w * 0.60), int(h * 0.43)), (int(w * 0.55), int(h * 0.435)), (int(w * 0.50), int(h * 0.43))], width_rel=0.0029, blur_rel=0.006)
    crow_L = line_mask(w, h, [(int(w * 0.50), int(h * 0.42)), (int(w * 0.52), int(h * 0.425)), (int(w * 0.535), int(h * 0.43))], width_rel=0.0024, blur_rel=0.010)
    crow_R = line_mask(w, h, [(int(w * 0.50), int(h * 0.42)), (int(w * 0.48), int(h * 0.425)), (int(w * 0.465), int(h * 0.43))], width_rel=0.0024, blur_rel=0.010)

    # Nose features
    nose_bridge = line_mask(w, h, [(int(w * 0.50), int(h * 0.38)), (int(w * 0.50), int(h * 0.52))], width_rel=0.0032, blur_rel=0.006)
    nose_tip_hi = ellipse_stroke_mask(w, h, int(w * 0.50), int(h * 0.55), int(w * 0.020), int(h * 0.012), width_rel=0.0030, blur_rel=0.007)
    nose_L = line_mask(w, h, [(int(w * 0.488), int(h * 0.47)), (int(w * 0.486), int(h * 0.53))], width_rel=0.0030, blur_rel=0.008)
    nose_R = line_mask(w, h, [(int(w * 0.512), int(h * 0.47)), (int(w * 0.514), int(h * 0.53))], width_rel=0.0030, blur_rel=0.008)

    # Lips contours
    upper_lip = line_mask(w, h, [(int(w * 0.44), int(h * 0.62)), (int(w * 0.50), int(h * 0.615)), (int(w * 0.56), int(h * 0.62))], width_rel=0.0042, blur_rel=0.009)
    lower_lip = line_mask(w, h, [(int(w * 0.44), int(h * 0.63)), (int(w * 0.50), int(h * 0.635)), (int(w * 0.56), int(h * 0.63))], width_rel=0.0038, blur_rel=0.009)
    cupid_bow = line_mask(w, h, [(int(w * 0.485), int(h * 0.615)), (int(w * 0.50), int(h * 0.610)), (int(w * 0.515), int(h * 0.615))], width_rel=0.0036, blur_rel=0.008)

    # Nasolabial folds and marionette lines
    naso_L = line_mask(w, h, [(int(w * 0.485), int(h * 0.56)), (int(w * 0.47), int(h * 0.60))], width_rel=0.0030, blur_rel=0.011)
    naso_R = line_mask(w, h, [(int(w * 0.515), int(h * 0.56)), (int(w * 0.53), int(h * 0.60))], width_rel=0.0030, blur_rel=0.011)
    marionette_L = line_mask(w, h, [(int(w * 0.47), int(h * 0.64)), (int(w * 0.47), int(h * 0.66))], width_rel=0.0028, blur_rel=0.011)
    marionette_R = line_mask(w, h, [(int(w * 0.53), int(h * 0.64)), (int(w * 0.53), int(h * 0.66))], width_rel=0.0028, blur_rel=0.011)

    # Under-chin line and neck shading
    under_chin = line_mask(w, h, [(int(w * 0.44), int(h * 0.655)), (int(w * 0.50), int(h * 0.66)), (int(w * 0.56), int(h * 0.655))], width_rel=0.0028, blur_rel=0.014)
    neck_vert_L = line_mask(w, h, [(int(w * 0.46), int(h * 0.67)), (int(w * 0.46), int(h * 0.64))], width_rel=0.0022, blur_rel=0.014)
    neck_vert_R = line_mask(w, h, [(int(w * 0.54), int(h * 0.67)), (int(w * 0.54), int(h * 0.64))], width_rel=0.0022, blur_rel=0.014)

    lines = union(
        oval, jaw,
        temple_L, temple_R,
        cheek_main_L, cheek_main_R, cheek_upper_L, cheek_upper_R, cheek_ridge_L, cheek_ridge_R, cheek_shadow_L, cheek_shadow_R,
        forehead_edge,
        upper_lid_L, lower_lid_L, upper_lid_R, lower_lid_R, crow_L, crow_R,
        nose_bridge, nose_tip_hi, nose_L, nose_R,
        upper_lip, lower_lip, cupid_bow, naso_L, naso_R, marionette_L, marionette_R,
        under_chin, neck_vert_L, neck_vert_R,
    )
    
    lines = ImageChops.multiply(lines, face)
    inv = ImageOps.invert(lines)
    contour = Image.composite(Image.new("L", (w, h), 0), white, lines)
    contour = contour.filter(ImageFilter.GaussianBlur(radius=0))

    contour_path = os.path.join(args.workdir, "a_pass", "contour_map.png")
    contour.save(contour_path)

    print("[A-PASS] Saved:", base_path, face_path, contour_path)


if __name__ == "__main__":
    main()
