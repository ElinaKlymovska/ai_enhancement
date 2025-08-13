
import os, requests
A1111_DIR = os.environ.get("A1111_DIR", "/workspace/stable-diffusion-webui")
MODELS_SD = os.path.join(A1111_DIR, "models", "Stable-diffusion")
MODELS_CN = os.path.join(A1111_DIR, "models", "ControlNet")
os.makedirs(MODELS_SD, exist_ok=True); os.makedirs(MODELS_CN, exist_ok=True)
CIV_BASE = "https://civitai.com/api/v1"
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY","")

def civ_get(path, params=None, stream=False):
    headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
    r = requests.get(f"{CIV_BASE}{path}", params=params, headers=headers, stream=stream, timeout=600)
    r.raise_for_status(); return r

def pick(ver):
    files = ver.get("files") or []
    def score(f):
        name=(f.get("name") or "").lower(); fmt=(f.get("format") or "").lower(); s=0
        if name.endswith(".safetensors"): s+=5
        if "sdxl" in name or "xl" in name: s+=3
        if "softedge" in name or "dexined" in name: s+=2
        if "canny" in name or "lineart" in name: s+=2
        if fmt=="safetensors": s+=1
        return s
    files.sort(key=score, reverse=True)
    return files[0] if files else None

def search_and_download(query, model_type, out_dir):
    print(f"[*] Search {model_type}: {query}")
    items = civ_get("/models", params={"query": query, "types": model_type, "limit": 10}).json().get("items") or []
    if not items: raise RuntimeError("No results")
    items.sort(key=lambda it:(it.get("stats",{}).get("downloadCount",0), it.get("modelVersions",[{}])[0].get("id",0)), reverse=True)
    ver = items[0].get("modelVersions",[{}])[0]
    f = pick(ver); url = f.get("downloadUrl"); name = f.get("name")
    if not url: raise RuntimeError("no downloadUrl")
    os.makedirs(out_dir, exist_ok=True); out = os.path.join(out_dir, name)
    if not os.path.exists(out):
        headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
        with requests.get(url, headers=headers, stream=True, timeout=1800) as rr:
            rr.raise_for_status()
            with open(out, "wb") as fp:
                for ch in rr.iter_content(1024*1024):
                    if ch: fp.write(ch)
    print("[OK]", out); return out

def main():
    try:
        sdxl = search_and_download(os.environ.get("SDXL_CIVITAI_QUERY","Realism Engine SDXL"), "Checkpoint", MODELS_SD)
        print("SDXL_CHECKPOINT=", os.path.basename(sdxl))
    except Exception as e:
        print("[WARN] SDXL:", e)
    # SoftEdge
    try:
        cn0 = search_and_download(os.environ.get("CONTROLNET_CIVITAI_QUERY","softedge sdxl dexined"), "Controlnet", MODELS_CN)
        print("CONTROLNET0_FILE=", os.path.basename(cn0))
    except Exception as e:
        print("[WARN] ControlNet0:", e)
    # Canny/Lineart
    try:
        cn1 = search_and_download(os.environ.get("CONTROLNET2_CIVITAI_QUERY","canny sdxl"), "Controlnet", MODELS_CN)
        print("CONTROLNET1_FILE=", os.path.basename(cn1))
    except Exception as e:
        print("[WARN] ControlNet1:", e)

if __name__ == "__main__":
    main()
