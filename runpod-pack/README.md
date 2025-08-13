# RunPod Pack — Headless A1111 + ADetailer + TWO ControlNets

**CN0:** SoftEdge SDXL (shape)  
**CN1:** Canny/Lineart (uses our `contour_map.png` from A-pass for crisp cheek/nose lines).

## Quick start
1) SSH to pod, upload the ZIP and unzip under `/workspace`
2) `cd /workspace/runpod-pack && cp .env.sample .env && bash bootstrap.sh`
3) Put inputs into `preset-contour/input/`

Tune `preset-contour/config.yaml -> b_pass`:
- `controlnet_weight` (0.55–0.65), `controlnet2_weight` (0.35–0.55)
- `ad_denoise` (0.14–0.20)
- `steps` 26–32, `cfg` 4.0–4.8
