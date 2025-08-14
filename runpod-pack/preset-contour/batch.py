
import os
import argparse
import subprocess
import sys
import yaml
import glob


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    input_dir = cfg["io"]["input_dir"]
    work_dir = cfg["io"]["work_dir"]
    out_dir = cfg["io"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        imgs += glob.glob(os.path.join(input_dir, ext))
    imgs.sort()
    
    if not imgs:
        print(f"No images in {input_dir}")
        return

    for i, img in enumerate(imgs, 1):
        print(f"\n=== [{i}/{len(imgs)}] {img} ===")
        name = os.path.splitext(os.path.basename(img))[0]
        wd = os.path.join(work_dir, name)
        os.makedirs(os.path.join(wd, "a_pass"), exist_ok=True)
        os.makedirs(os.path.join(wd, "masks"), exist_ok=True)

        subprocess.run([sys.executable, "run_a_pass.py", "--config", args.config, "--input", img, "--workdir", wd], check=True)
        subprocess.run([sys.executable, "run_b_pass.py", "--config", args.config, "--workdir", wd, "--output", out_dir], check=True)

    print("\nAll done.")


if __name__ == "__main__":
    main()
