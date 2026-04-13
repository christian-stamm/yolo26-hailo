import argparse
import concurrent.futures
import json
import os
import random
import shutil
import tempfile
import urllib.request
import zipfile


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_TRAIN_IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"
DEFAULT_OUTPUT_DIR = "/workspaces/yolo26-hailo/res/datasets/calib_images"
DEFAULT_CACHE_DIR = "/workspaces/yolo26-hailo/res/datasets/coco"


def _download_file(url: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, destination)


def _load_train_image_file_names(cache_dir: str) -> list[str]:
    annotations_zip = os.path.join(cache_dir, "annotations_trainval2017.zip")
    annotations_dir = os.path.join(cache_dir, "annotations")
    train_json_path = os.path.join(annotations_dir, "instances_train2017.json")

    _download_file(COCO_ANNOTATIONS_URL, annotations_zip)

    if not os.path.exists(train_json_path):
        print("Extracting COCO annotations...")
        os.makedirs(annotations_dir, exist_ok=True)
        with zipfile.ZipFile(annotations_zip, "r") as zip_ref:
            zip_ref.extractall(cache_dir)

    with open(train_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    images = payload.get("images", [])
    if not images:
        raise RuntimeError(f"No images found in {train_json_path}")

    return [img["file_name"] for img in images if "file_name" in img]


def _download_one_image(file_name: str, output_dir: str) -> bool:
    out_path = os.path.join(output_dir, file_name)
    if os.path.exists(out_path):
        return True

    url = f"{COCO_TRAIN_IMAGE_BASE_URL}/{file_name}"
    temp_path = ""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            # Keep temp file on the same filesystem as destination for atomic replace.
            with tempfile.NamedTemporaryFile(delete=False, dir=output_dir) as tmp:
                shutil.copyfileobj(response, tmp)
                temp_path = tmp.name
        os.replace(temp_path, out_path)
        return True
    except Exception as exc:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        print(f"Failed downloading {file_name}: {exc}")
        return False


def download_hailo_calibration_set(
    output_dir: str,
    num_samples: int,
    cache_dir: str,
    seed: int,
    workers: int,
) -> None:
    """Download a random subset of COCO train2017 images for calibration."""
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading COCO train image list from {cache_dir}...")
    file_names = _load_train_image_file_names(cache_dir)

    if num_samples > len(file_names):
        raise ValueError(
            f"Requested {num_samples} samples, but only {len(file_names)} available"
        )

    rng = random.Random(seed)
    selected = rng.sample(file_names, num_samples)

    print(
        f"Downloading {num_samples} calibration images into {output_dir} "
        f"with {workers} workers..."
    )
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_one_image, name, output_dir) for name in selected]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            if future.result():
                success += 1
            if idx % 50 == 0 or idx == num_samples:
                print(f"Progress: {idx}/{num_samples}, success: {success}")

    if success != num_samples:
        raise RuntimeError(
            f"Completed with missing files: requested={num_samples}, downloaded={success}"
        )

    print(f"Successfully downloaded {success} calibration images to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO calibration images")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--samples", type=int, default=1024, help="Number of samples")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory used to cache COCO annotation files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    args = parser.parse_args()

    download_hailo_calibration_set(
        output_dir=args.output,
        num_samples=args.samples,
        cache_dir=args.cache_dir,
        seed=args.seed,
        workers=args.workers,
    )
