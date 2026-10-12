import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",
}


def _detect_image_ext(data: bytes) -> str:
    for sig, ext in IMAGE_SIGNATURES.items():
        if data[: len(sig)] == sig:
            return ext
    return "jpg"


_CDN_HOST_MAP = {
    "ecx.images-amazon.com": "m.media-amazon.com",
    "g-ecx.images-amazon.com": "m.media-amazon.com",
    "images.amazon.com": "m.media-amazon.com",
    "z-ecx.images-amazon.com": "m.media-amazon.com",
}

_SIZE_SUFFIX_RE = re.compile(r"(/I/[A-Za-z0-9+\-_]+)\.[^/]*?(\.[A-Za-z]{3,4})$")


def normalize_imurl(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url)
    new_host = _CDN_HOST_MAP.get(parts.netloc.lower())
    if not new_host:
        return url
    return urlunsplit(("https", new_host, parts.path, parts.query, parts.fragment))


def _strip_size_suffix(url: str) -> str:
    if not url:
        return ""
    m = _SIZE_SUFFIX_RE.search(url)
    if not m:
        return ""
    return url[: m.start()] + m.group(1) + m.group(2)


def _candidate_urls(url: str) -> list:
    candidates = []
    normalized = normalize_imurl(url)
    if normalized:
        candidates.append(normalized)
    stripped = _strip_size_suffix(normalized or url)
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    return candidates or [url]


def download_single_image(
    url: str,
    save_dir: str,
    item_id: str,
    max_retries: int = 5,
    timeout: int = 15,
    retry_delay: float = 2.0,
) -> Tuple[str, bool, str]:
    if not url:
        return item_id, False, "URL is empty"

    last_error = "unknown error"
    candidates = _candidate_urls(url)
    for cand in candidates:
        for attempt in range(max_retries):
            try:
                req = Request(cand, headers={"User-Agent": "Mozilla/5.0"})

                with urlopen(req, timeout=timeout) as response:
                    if response.status != 200:
                        last_error = f"HTTP {response.status} ({cand})"
                        break

                    data = response.read()

                    if len(data) < 100:
                        last_error = f"file too small ({len(data)} bytes), probably not an image"
                        break

                    ext = _detect_image_ext(data)
                    save_path = os.path.join(save_dir, f"{item_id}.{ext}")

                    with open(save_path, "wb") as f:
                        f.write(data)

                    return item_id, True, f"downloaded ({len(data)} bytes)"

            except HTTPError as e:
                last_error = f"HTTP {e.code} ({cand})"
                break
            except (URLError, TimeoutError, OSError) as e:
                last_error = f"{e}"
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2**attempt))

    return item_id, False, f"failed after {max_retries} retries: {last_error}"


def download_all_images(
    meta_dict: Dict[str, dict],
    save_dir: str,
    max_workers: int = 4,
    max_retries: int = 5,
    timeout: int = 15,
    retry_delay: float = 2.0,
) -> Dict[str, dict]:
    os.makedirs(save_dir, exist_ok=True)

    to_download = []
    skipped = 0
    rewritten = 0
    for asin, info in meta_dict.items():
        exists = False
        for ext in ["jpg", "png", "gif", "webp"]:
            if os.path.exists(os.path.join(save_dir, f"{asin}.{ext}")):
                exists = True
                break
        if exists:
            skipped += 1
            continue

        url = info.get("imUrl", "")
        if url:
            if normalize_imurl(url) != url:
                rewritten += 1
            to_download.append((asin, url))

    logger.info(
        f"to download: {len(to_download)}, already present (skipped): {skipped}"
    )
    if rewritten:
        logger.info(
            f"URL normalization: {rewritten} legacy domains (ecx.images-amazon.com etc.) -> m.media-amazon.com"
            f", with a suffix-stripping fallback for size-suffixed URLs"
        )

    if not to_download:
        return {
            "total": len(meta_dict),
            "success": skipped,
            "failed": 0,
            "skipped": skipped,
        }

    success_count = 0
    failed_count = 0
    failed_items = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_single_image,
                url,
                save_dir,
                asin,
                max_retries,
                timeout,
                retry_delay,
            ): asin
            for asin, url in to_download
        }

        try:
            from tqdm import tqdm

            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Download images",
                ncols=80,
            )
        except ImportError:
            progress = as_completed(futures)

        for future in progress:
            item_id, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_items.append((item_id, message))
                if failed_count <= 10:
                    logger.warning(f"Download failed [{item_id}]: {message}")

    if failed_items:
        failed_path = os.path.join(save_dir, "download_failed.txt")
        with open(failed_path, "w", encoding="utf-8") as f:
            for item_id, reason in failed_items:
                f.write(f"{item_id}\t{reason}\n")
        logger.info(f"Failed list saved to: {failed_path}")

    stats = {
        "total": len(meta_dict),
        "success": success_count + skipped,
        "failed": failed_count,
        "skipped": skipped,
        "downloaded": success_count,
    }

    logger.info(
        f"Download done: success {stats['success']}, failed {failed_count}, skipped {skipped}"
    )
    return stats
