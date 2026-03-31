import argparse
import contextlib
import gc
import os
import sys

import h5py
import torch
from huggingface_hub import login
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

MODEL_ID = "MahmoodLab/TITAN"
CONCH_FEATURES_KEY = "conch15/features"
COORDS_KEY = "coordinates"
TITAN_FEATURES_KEY = "titan/features"
PATCH_SIZE_LEVEL0 = 256
# True: CUDA が2枚以上のとき nn.DataParallel でラップ（効かない場合が多い）
USE_DATAPARALLEL = True


def _encode_slide_from_patch_features(model, features, coords, patch_size_level0):
    """DataParallel ラップ時は module 側のメソッドを呼ぶ（ラッパーに同名が無いため）。"""
    inner = model.module if isinstance(model, torch.nn.DataParallel) else model
    return inner.encode_slide_from_patch_features(
        features, coords, patch_size_level0
    )


def extract_titan_slide_embedding(
    model: torch.nn.Module,
    h5_path: str,
    conch_features_key: str = CONCH_FEATURES_KEY,
    coords_key: str = COORDS_KEY,
    titan_features_key: str = TITAN_FEATURES_KEY,
):
    logger.info(f"Processing H5 file: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if conch_features_key not in f:
            raise KeyError(f"Missing dataset: {conch_features_key}")
        if coords_key not in f:
            raise KeyError(f"Missing dataset: {coords_key}")

        features_np = f[conch_features_key][:]
        coords_np = f[coords_key][:]

    logger.info(f"Loaded conch features shape: {features_np.shape}")
    logger.info(f"Loaded coords shape: {coords_np.shape}")
    logger.info(f"patch_size_level0: {PATCH_SIZE_LEVEL0}")

    features = torch.from_numpy(features_np).to(device, non_blocking=True)
    coords = torch.from_numpy(coords_np).long().to(device, non_blocking=True)

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    with autocast_ctx, torch.inference_mode():
        slide_embedding = _encode_slide_from_patch_features(
            model, features, coords, PATCH_SIZE_LEVEL0
        )

    logger.info(f"Slide embedding shape: {slide_embedding.shape}")

    slide_emb_np = slide_embedding.detach().cpu().numpy()

    group_name, dataset_name = titan_features_key.rsplit("/", 1)
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group(group_name)
        if dataset_name in grp:
            logger.info(f"Overwriting existing dataset: {titan_features_key}")
            del grp[dataset_name]
        grp.create_dataset(
            dataset_name,
            data=slide_emb_np,
            compression="gzip",
        )

    logger.info(f"Saved {titan_features_key} with shape {slide_emb_np.shape}")

    del features, coords, slide_embedding, slide_emb_np
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def load_titan_model():
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    # DataParallel は forward のバッチ分割向け。TITAN のスライドエンコードは
    # 全パッチが相互作用するため、これで VRAM が割れるとは限らない（効かないことが多い）。
    if (
        USE_DATAPARALLEL
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        logger.warning(
            "USE_DATAPARALLEL: nn.DataParallel を有効化。"
        )
        model = torch.nn.DataParallel(model)
    return model


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Process all .h5 files in a directory: add TITAN slide embeddings from ConCH patch features."
    )
    p.add_argument(
        "directory",
        help="Directory containing .h5 files",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.directory

    if not os.path.isdir(input_dir):
        logger.error(f"Not a directory: {input_dir}")
        return 1

    token = os.environ.get("HF_TOKEN")
    login(token=token) if token else login()

    model = load_titan_model()

    h5_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".h5")
    )
    logger.info(f"Found {len(h5_files)} h5 files in directory: {input_dir}")

    for h5_path in tqdm(h5_files):
        try:
            with h5py.File(h5_path, "r") as f:
                if TITAN_FEATURES_KEY in f:
                    logger.info(
                        f"Skipping {h5_path}: '{TITAN_FEATURES_KEY}' already exists."
                    )
                    print(f"[SKIP] {h5_path}: '{TITAN_FEATURES_KEY}' already exists.")
                    continue

            extract_titan_slide_embedding(model, h5_path=h5_path)
            print(f"[OK] {h5_path}")
        except Exception as e:
            logger.exception(f"Failed to process H5 file: {h5_path} error={e}")
            print(f"[NG] {h5_path} error={e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# TITAN_DATAPARALLEL=1 uv run python extract_feratures.py /path/to/h5_directory