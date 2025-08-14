
import os
import argparse
import subprocess
import sys
import yaml
import glob
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Налаштування логування з кольоровими повідомленнями."""
    logger = logging.getLogger("ADetailerBatch")
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_cfg(path: str) -> Dict[str, Any]:
    """Завантаження конфігурації з валідацією."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Базова валідація
        required_keys = ["io", "b_pass"]
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required config key: {key}")
        
        return cfg
    except Exception as e:
        raise RuntimeError(f"Failed to load config {path}: {e}")


def process_single_image(img_path: str, work_dir: str, config_path: str, logger: logging.Logger) -> bool:
    """Обробка одного зображення з retry механізмом."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Processing {os.path.basename(img_path)} (attempt {attempt + 1}/{max_retries})")
            
            # Створення робочої директорії
            name = os.path.splitext(os.path.basename(img_path))[0]
            wd = os.path.join(work_dir, name)
            os.makedirs(os.path.join(wd, "a_pass"), exist_ok=True)
            os.makedirs(os.path.join(wd, "masks"), exist_ok=True)
            
            # A-Pass
            logger.debug(f"Running A-Pass for {name}")
            result_a = subprocess.run(
                [sys.executable, "run_a_pass.py", "--config", config_path, "--input", img_path, "--workdir", wd],
                capture_output=True, text=True, check=True
            )
            
            # B-Pass
            logger.debug(f"Running B-Pass for {name}")
            result_b = subprocess.run(
                [sys.executable, "run_b_pass.py", "--config", config_path, "--workdir", wd, "--output", "output"],
                capture_output=True, text=True, check=True
            )
            
            logger.info(f"Successfully processed {name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Process failed for {os.path.basename(img_path)} (attempt {attempt + 1}): {e}")
            if e.stdout:
                logger.debug(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.debug(f"STDERR: {e.stderr}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to process {os.path.basename(img_path)} after {max_retries} attempts")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error processing {os.path.basename(img_path)}: {e}")
            return False
    
    return False


def main():
    ap = argparse.ArgumentParser(description="Batch processing for ADetailer + ControlNet")
    ap.add_argument("--config", default="config.yaml", help="Path to config file")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = ap.parse_args()
    
    # Налаштування логування
    logger = setup_logging(args.log_level)
    logger.info("Starting ADetailer batch processing")
    
    try:
        cfg = load_cfg(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    input_dir = cfg["io"]["input_dir"]
    work_dir = cfg["io"]["work_dir"]
    out_dir = cfg["io"]["output_dir"]
    
    # Створення директорій
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    
    # Пошук зображень
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        imgs += glob.glob(os.path.join(input_dir, ext))
    imgs.sort()
    
    if not imgs:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(imgs)} images to process")
    
    # Обробка зображень
    successful = 0
    failed = 0
    
    if args.workers == 1:
        # Послідовна обробка з прогрес-баром
        for img in tqdm.tqdm(imgs, desc="Processing images", unit="img"):
            if process_single_image(img, work_dir, args.config, logger):
                successful += 1
            else:
                failed += 1
    else:
        # Паралельна обробка
        logger.info(f"Using {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Створення футур для всіх зображень
            future_to_img = {
                executor.submit(process_single_image, img, work_dir, args.config, logger): img 
                for img in imgs
            }
            
            # Обробка завершених футур з прогрес-баром
            with tqdm.tqdm(total=len(imgs), desc="Processing images", unit="img") as pbar:
                for future in as_completed(future_to_img):
                    img = future_to_img[future]
                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Exception occurred while processing {os.path.basename(img)}: {e}")
                        failed += 1
                    pbar.update(1)
    
    # Підсумок
    logger.info(f"Processing completed!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {len(imgs)}")
    
    if failed > 0:
        logger.warning(f"{failed} images failed to process. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All images processed successfully!")


if __name__ == "__main__":
    main()
